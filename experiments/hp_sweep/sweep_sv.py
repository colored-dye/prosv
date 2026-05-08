from collections import defaultdict
from dataclasses import dataclass
import itertools
import json
from loguru import logger
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Literal, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    set_seed,
    get_scheduler,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.hf_argparser import HfArg

from reft.constants import IGNORE_INDEX
from reft.utils import (
    load_hf_model_tokenizer,
    set_decoder_norm_to_unit_norm,
    get_batch_logps,
    preference_loss,
)
from reft.interventions import (
    ClampFreeIntervention,
    AdditionFreeIntervention,
    LoreftAdapter,
    PreferenceIntervention,
)
from reft.dataset import curate_training_data, curate_preference_training_data
from reft.intervenable import (
    IntervenableModel,
    IntervenableConfig,
    RepresentationConfig,
)


ADAPTER_CLASS_MAP = {
    "clamp_free": ClampFreeIntervention,
    "add_free": AdditionFreeIntervention,
    "loreft": LoreftAdapter,
    "pref": PreferenceIntervention,
}


@dataclass
class Arguments:
    seed: int = HfArg(default=42)

    model_path: str = HfArg(default="google/gemma-2-2b-it")
    concept_data_dir: str = HfArg(default="prod_2b_l10_v1/generate")
    output_dir: str = HfArg(default="outputs/")
    layers: List[int] = HfArg(default_factory=list)

    low_rank_dim: int = HfArg(default=1)
    alpha: float = HfArg(default=1.0)
    adapter_type: str = HfArg(default="dora")
    positions: str = HfArg(default="f4")
    optimizer: Literal["adam", "sgd"] = HfArg(default="adam")
    objective: Literal["lang", "simpo", "bisimpo"] = HfArg(default="lang")
    max_output_length: int = HfArg(default=128)
    prefix_length: int = HfArg(default=None)
    gradient_accumulation_steps: int = HfArg(default=1)

    batch_size_list: List[int] = HfArg(default_factory=list)
    epochs_list: List[int] = HfArg(default_factory=list)
    vector_learning_rate_list: List[str] = HfArg(default_factory=list)
    vector_init_scale_list: List[str] = HfArg(default_factory=list)
    factor_learning_rate_list: List[str] = HfArg(default_factory=list)
    factor_init_scale_list: List[str] = HfArg(default_factory=list)


def train_sv(
    args: Arguments,
    seed: int,
    epochs: int,
    factor_init_scale: float,
    vector_init_scale: float,
    factor_learning_rate: float,
    vector_learning_rate: float,
    model: PreTrainedModel,
    adapter_class,
    train_dataloader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    concept: str,
    concept_id: int,
    do_save: bool = True,
):
    set_seed(seed)

    embed_dim = model.config.hidden_size

    rep_configs = []
    for layer_i in args.layers:
        module_name = f"model.layers.{layer_i}"
        rep_cfg = RepresentationConfig(
            layer=layer_i,
            embed_dim=embed_dim,
            low_rank_dim=1,
            target_module=module_name,
            intervention_type=adapter_class.__name__,
            factor_init_scale=factor_init_scale,
            vector_init_scale=vector_init_scale,
        )
        rep_configs.append(rep_cfg)
    intervenable_config = IntervenableConfig(representations=rep_configs)
    intervenable = IntervenableModel(model=model, config=intervenable_config)

    # trainable_params = intervenable.get_trainable_params()
    factor_params, vector_params = intervenable.get_trainable_params(separate=True)

    param_groups = []
    for _, _lora in intervenable.interventions.items():
        match args.adapter_type:
            case "loreft" | "direft_unit":
                param_groups.append(
                    {
                        "params": _lora.rotate_layer.parameters(),
                        "lr": vector_learning_rate,
                    }
                )
                param_groups.append(
                    {"params": _lora.learned_source.weight, "lr": vector_learning_rate}
                )
                param_groups.append(
                    {"params": _lora.learned_source.bias, "lr": vector_learning_rate}
                )
            case "bilin":
                param_groups.append(
                    {"params": _lora.parameters(), "lr": vector_learning_rate}
                )
            case _:
                factor_lr = factor_learning_rate
                logger.warning(
                    f"Factor LR: {factor_lr:.3f} || Factor init scale: {factor_init_scale:.3f}"
                )

                param_groups.append(
                    {"params": _lora.proj.weight, "lr": vector_learning_rate}
                )
                param_groups.append({"params": _lora.factor, "lr": factor_lr})

    num_trainable_params, ratio_trainable_params = (
        intervenable.get_num_trainable_params()
    )
    logger.warning(
        f"Trainable parameters: {num_trainable_params} || {ratio_trainable_params * 100:.3e}%"
    )

    match args.optimizer:
        case "adam":
            optimizer = torch.optim.Adam(param_groups)
        case "sgd":
            optimizer = torch.optim.SGD(param_groups)
        case _:
            raise ValueError(f"Unknown optimizer: `{args.optimizer}`")

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_training_steps=epochs * len(train_dataloader),
        num_warmup_steps=0,
    )

    logger.warning(f"[Concept {concept_id}]: {concept}")

    all_step_loss = []
    all_epoch_loss = []
    all_step_factor = []
    pgbar_epoch = tqdm(range(epochs), desc=f"Training on concept [{concept_id}]")
    # CLIP_GRAD_NORM = None
    for epoch_i in pgbar_epoch:
        epoch_loss = 0
        epoch_reward_acc = 0

        pgbar_step = tqdm(
            train_dataloader, desc=f"Epoch [{epoch_i + 1}/{epochs}]", disable=True
        )

        for step, batch in enumerate(pgbar_step):
            if args.objective == "lang":
                locations = batch["intervention_locations"]

                outputs = intervenable(
                    locations=locations,
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                logits = outputs.logits[:, :-1].contiguous()
                shift_logits = logits.view(-1, logits.size(-1))
                labels = batch["labels"][:, 1:].contiguous().to(device)
                shift_labels = labels.view(-1)
                loss = nn.functional.cross_entropy(
                    shift_logits, shift_labels, reduction="mean"
                )
                loss.backward()
                epoch_loss += loss.item()
            elif args.objective == "simpo":
                batch_processed = {}
                # winning always before losing
                # https://github.com/stanfordnlp/axbench/blob/main/axbench/models/preference_model.py#L323
                for item in ("input_ids", "labels", "attention_mask", "intervention_locations"):
                    batch_processed[item] = torch.cat([batch[f"positive_{item}"], batch[f"negative_{item}"]], dim=0)

                locations = batch_processed["intervention_locations"]
                policy_outputs = intervenable(
                    locations=locations,
                    input_ids=batch_processed["input_ids"].to(device),
                    attention_mask=batch_processed["attention_mask"].to(device),
                )
                labels = batch_processed["labels"].to(device)
                policy_logps = get_batch_logps(policy_outputs.logits, labels, average_log_prob=False)

                ref_outputs = intervenable.model(
                    input_ids=batch_processed["input_ids"].to(device),
                    attention_mask=batch_processed["attention_mask"].to(device),
                )
                ref_logps = get_batch_logps(ref_outputs.logits, labels, average_log_prob=False)

                cur_batch_size = batch_processed["input_ids"].size(0)
                winning_logps = policy_logps[:cur_batch_size//2]
                losing_logps = policy_logps[cur_batch_size//2:]
                ref_winning_logps = ref_logps[:cur_batch_size//2]
                ref_losing_logps = ref_logps[cur_batch_size//2:]

                def get_lens(labels):
                    mask = labels != IGNORE_INDEX
                    return mask.sum(dim=-1)
                winning_lens = get_lens(batch_processed["labels"][:cur_batch_size//2]).to(device)
                losing_lens = get_lens(batch_processed["labels"][cur_batch_size//2:]).to(device)

                # beta, gemma, simpo_scaler is set based on:
                # https://github.com/stanfordnlp/axbench/blob/main/axbench/sweep/wuzhengx/reps/experiments/p_vector_dps_g2-2b_axbench.yaml
                batch_preference_losses, chosen_rewards, rejected_rewards = preference_loss(
                    policy_chosen_logps=winning_logps,
                    policy_rejected_logps=losing_logps,
                    reference_chosen_logps=ref_winning_logps,
                    reference_rejected_logps=ref_losing_logps,
                    beta=1.0,
                    gemma=0.0,
                    simpo_scaler=1.0,
                    winning_lens=winning_lens,
                    losing_lens=losing_lens,
                    loss_type="scaled_simpo",
                )
                loss = batch_preference_losses.mean()
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                epoch_loss += loss.item()
            elif args.objective == "bisimpo":
                chosen_rewards = []
                rejected_rewards = []

                for mode in ("add", "clamp"):
                    batch_processed = {}
                    # winning always before losing
                    # https://github.com/stanfordnlp/axbench/blob/main/axbench/models/preference_model.py#L323
                    for item in ("input_ids", "labels", "attention_mask", "intervention_locations"):
                        if mode == "add":
                            batch_processed[item] = torch.cat([batch[f"positive_{item}"], batch[f"negative_{item}"]], dim=0)
                        else:
                            batch_processed[item] = torch.cat([batch[f"negative_{item}"], batch[f"positive_{item}"]], dim=0)

                    for k, intv in intervenable.interventions.items():
                        if isinstance(intv, PreferenceIntervention):
                            intv.mode = mode

                    locations = batch_processed["intervention_locations"]
                    policy_outputs = intervenable(
                        locations=locations,
                        input_ids=batch_processed["input_ids"].to(device),
                        attention_mask=batch_processed["attention_mask"].to(device),
                    )
                    labels = batch_processed["labels"].to(device)
                    policy_logps = get_batch_logps(policy_outputs.logits, labels, average_log_prob=False)

                    ref_outputs = intervenable.model(
                        input_ids=batch_processed["input_ids"].to(device),
                        attention_mask=batch_processed["attention_mask"].to(device),
                    )
                    ref_logps = get_batch_logps(ref_outputs.logits, labels, average_log_prob=False)

                    cur_batch_size = batch_processed["input_ids"].size(0)
                    winning_logps = policy_logps[:cur_batch_size//2]
                    losing_logps = policy_logps[cur_batch_size//2:]
                    ref_winning_logps = ref_logps[:cur_batch_size//2]
                    ref_losing_logps = ref_logps[cur_batch_size//2:]

                    def get_lens(labels):
                        mask = labels != IGNORE_INDEX
                        return mask.sum(dim=-1)
                    winning_lens = get_lens(batch_processed["labels"][:cur_batch_size//2]).to(device)
                    losing_lens = get_lens(batch_processed["labels"][cur_batch_size//2:]).to(device)

                    # beta, gemma, simpo_scaler is set based on:
                    # https://github.com/stanfordnlp/axbench/blob/main/axbench/sweep/wuzhengx/reps/experiments/p_vector_dps_g2-2b_axbench.yaml
                    batch_preference_losses, batch_chosen_rewards, batch_rejected_rewards = preference_loss(
                        policy_chosen_logps=winning_logps,
                        policy_rejected_logps=losing_logps,
                        reference_chosen_logps=ref_winning_logps,
                        reference_rejected_logps=ref_losing_logps,
                        beta=1.0,
                        gemma=0.0,
                        simpo_scaler=1.0,
                        winning_lens=winning_lens,
                        losing_lens=losing_lens,
                        loss_type="scaled_simpo",
                    )
                    loss = batch_preference_losses.mean()
                    loss = loss / (2 * args.gradient_accumulation_steps) # since we split two minibatches
                    loss.backward()

                    chosen_rewards.append(batch_chosen_rewards.detach())
                    rejected_rewards.append(batch_rejected_rewards.detach())
                    epoch_loss += loss.item()

                chosen_rewards = torch.cat(chosen_rewards)
                rejected_rewards = torch.cat(rejected_rewards)
            else:
                raise ValueError(f"Unknown objective: `{args.objective}`")

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping should not affect theoretical assumptions.
                # Clip all gradients as a single tensor.
                # This approach is wrong!!!
                # torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

                # Only clip vector gradients
                torch.nn.utils.clip_grad_norm_(vector_params, 1.0)
                # if CLIP_GRAD_NORM is None:
                #     CLIP_GRAD_NORM = True
                    # logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # logger.critical("Using gradient clipping!")
                    # logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!")

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if args.adapter_type == "add_unit" or args.adapter_type == "clamp_unit":
                    for _, _lora in intervenable.interventions.items():
                        set_decoder_norm_to_unit_norm(_lora.proj)

            pgbar_log = {
                "loss": f"{loss.item():.4f}",
            }
            if args.adapter_type != "bilin":
                factor = (
                    _lora.factor.item()
                    if hasattr(_lora, "factor")
                    else _lora.learned_source.bias[0].item()
                )
                l2_norm = (
                    _lora.proj.weight.data.norm()
                    if hasattr(_lora, "proj")
                    else _lora.learned_source.weight.data.norm()
                )
                pgbar_log.update(
                    {"factor": f"{factor:.2f}", "l2_norm": f"{l2_norm:.4f}"}
                )
                all_step_factor.append(factor)
            if args.objective == "simpo" or args.objective == "bisimpo":
                reward_accuracies = (chosen_rewards > rejected_rewards).float().mean().cpu().numpy()
                epoch_reward_acc += reward_accuracies
            pgbar_step.set_postfix(pgbar_log)

            all_step_loss.append(loss.item())
            intervenable.clear_cache()

        pgbar_step.close()

        epoch_loss /= len(train_dataloader)
        all_epoch_loss.append(epoch_loss)
        pgbar_log = {
            "loss": f"{epoch_loss:.4f}",
        }
        if args.adapter_type != "bilin":
            factor = (
                _lora.factor.item()
                if hasattr(_lora, "factor")
                else _lora.learned_source.bias[0].item()
            )
            l2_norm = (
                _lora.proj.weight.data.norm()
                if hasattr(_lora, "proj")
                else _lora.learned_source.weight.data.norm()
            )
            pgbar_log.update({"factor": f"{factor:.2f}", "l2_norm": f"{l2_norm:.4f}"})
        if args.objective == "simpo" or args.objective == "bisimpo":
            epoch_reward_acc /= len(train_dataloader)
            pgbar_log.update({"reward_acc": f"{epoch_reward_acc:.2f}"})
        pgbar_epoch.set_postfix(pgbar_log)

    pgbar_epoch.close()

    if do_save:
        save_dir = Path(args.output_dir) / f"{concept_id}"
        save_dir.mkdir(parents=True, exist_ok=True)

        intervenable.save(save_dir)

        # log = {
        #     "losses_epoch": all_epoch_loss,
        #     "losses_step": all_step_loss,
        #     "factor_step": all_step_factor,
        # }
        # save_path = save_dir / "log.pt"
        # torch.save(log, save_path)
        # logger.warning(f"Saved to `{save_path}`")


def get_concept_dataloader(
    args: Arguments,
    tokenizer: PreTrainedTokenizer,
    concept_id: int,
    batch_size: int,
):
    if args.objective == "lang":
        data_path = Path(args.concept_data_dir, "train_data.parquet")
        df = pd.read_parquet(data_path)

        concept_df = df[df["concept_id"] == concept_id]

        dataset = []
        for _, row in concept_df.iterrows():
            dataset.append((row["input"], row["output"]))

        inputs = []
        outputs = []
        for x in dataset:
            ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": x[0]}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            if (tokenizer.bos_token is not None) and (ids[0] == tokenizer.bos_token_id):
                ids = ids[1:]
            inputs.append(tokenizer.decode(ids))

            output_ids = tokenizer.encode(x[1])
            if (tokenizer.bos_token is not None) and (output_ids[0] == tokenizer.bos_token_id):
                output_ids = output_ids[1:]
            outputs.append(tokenizer.decode(output_ids))

        if args.prefix_length is not None:
            logger.critical(f"Intervention location skips prefix {args.prefix_length}")

        data_module = curate_training_data(
            tokenizer=tokenizer,
            positions=args.positions,
            inputs=inputs,
            outputs=outputs,
            padding_side="right",
            prefix_length=args.prefix_length,
        )
    elif args.objective == "simpo" or args.objective == "bisimpo":
        data_path = Path(args.concept_data_dir, "contrast_train_data.parquet")
        df = pd.read_parquet(data_path)

        concept_df = df[df["concept_id"] == concept_id]
        dataset = []
        # We use negative inputs for default inputs.
        for _, row in concept_df.iterrows():
            dataset.append((row["negative_input"], row["positive_output"], row["negative_output"]))

        inputs = []
        positive_outputs = []
        negative_outputs = []
        for x in dataset:
            ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": x[0]}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            if (tokenizer.bos_token is not None) and (ids[0] == tokenizer.bos_token_id):
                ids = ids[1:]
            inputs.append(tokenizer.decode(ids))

            pos_output_ids = tokenizer.encode(x[1])
            neg_output_ids = tokenizer.encode(x[2])
            if (tokenizer.bos_token is not None) and (pos_output_ids[0] == tokenizer.bos_token_id):
                pos_output_ids = pos_output_ids[1:]
                neg_output_ids = neg_output_ids[1:]
            positive_outputs.append(
                tokenizer.decode(pos_output_ids[: args.max_output_length])
            )
            negative_outputs.append(
                tokenizer.decode(neg_output_ids[: args.max_output_length])
            )

        if args.prefix_length is not None:
            logger.critical(f"Intervention location skips prefix {args.prefix_length}")
        data_module = curate_preference_training_data(
            tokenizer=tokenizer,
            positions=args.positions,
            positive_inputs=inputs,
            negative_inputs=inputs,
            positive_outputs=positive_outputs,
            negative_outputs=negative_outputs,
            padding_side="right",
            prefix_length=args.prefix_length,
        )
    else:
        raise ValueError(f"Invalid objective: {args.objective}")

    train_set, collator = data_module["train_dataset"], data_module["data_collator"]

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataloader = DataLoader(
        train_set,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )

    if args.objective == "lang":
        logger.warning("**" + tokenizer.decode(train_set[0]["input_ids"]) + "**")
    elif args.objective == "simpo" or args.objective == "bisimpo":
        logger.warning("Positive: **" + tokenizer.decode(train_set[0]["positive_input_ids"]) + "**")
        logger.warning("Negative: **" + tokenizer.decode(train_set[0]["negative_input_ids"]) + "**")

    return train_dataloader


def main(args: Arguments):
    logger.warning(args)
    set_seed(args.seed)

    device = "cuda"
    dtype = torch.bfloat16
    model, tokenizer = load_hf_model_tokenizer(
        model_name_or_path=args.model_path,
        device=device,
        dtype=dtype,
        padding_side="right",
    )

    adapter_class = ADAPTER_CLASS_MAP.get(args.adapter_type)
    if adapter_class is None:
        raise ValueError(f"Unknown adapter type: `{args.adapter_type}`")

    metadata_path = Path(args.concept_data_dir, "metadata.jsonl")
    metadata = defaultdict()
    for line in open(metadata_path).readlines():
        rec = json.loads(line)
        metadata[rec["concept_id"]] = rec

    output_base_dir = args.output_dir

    for batch_size, epochs, vector_lr, vector_init, factor_lr, factor_init in itertools.product(
        args.batch_size_list,
        args.epochs_list,
        args.vector_learning_rate_list,
        args.vector_init_scale_list,
        args.factor_learning_rate_list,
        args.factor_init_scale_list,
    ):
        logger.warning("============================================================================")
        logger.warning(
            f"B: {batch_size} || E: {epochs} || F_lr: {factor_lr} || F_init: {factor_init} || V_lr: {vector_lr} || V_init: {vector_init}"
        )
        logger.warning("============================================================================")

        args.output_dir = Path(
            output_base_dir,
            f"b={batch_size}_e={epochs}_vlr={vector_lr}_vinit={vector_init}_flr={factor_lr}_finit={factor_init}"
        )

        for concept_id, metarec in metadata.items():
            save_dir = Path(args.output_dir) / f"{concept_id}"
            if (save_dir / "state_dict.pt").exists():
                logger.warning(f"Already done concept id: {concept_id}; skipping")
                continue
            
            train_dataloader = get_concept_dataloader(
                args, tokenizer, concept_id, batch_size,
            )
            concept = metarec["concept"]

            train_sv(
                args=args,
                model=model,
                adapter_class=adapter_class,
                train_dataloader=train_dataloader,
                device=device,
                dtype=dtype,
                concept=concept,
                concept_id=concept_id,
                epochs=int(epochs),
                vector_learning_rate=float(vector_lr),
                vector_init_scale=float(vector_init),
                factor_init_scale=float(factor_init),
                factor_learning_rate=float(factor_lr),
                seed=args.seed,
            )


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    main(parser.parse_args())
