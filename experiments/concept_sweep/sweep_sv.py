from collections import defaultdict
from dataclasses import dataclass
import json
from loguru import logger
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Literal

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

from reft.utils import load_hf_model_tokenizer
from reft.interventions import (
    RepresentationDoRA,
    RepresentationLoRA,
    AdditionUnitIntervention,
    ClampUnitIntervention,
    ClampFreeIntervention,
    AdditionFreeIntervention,
    LoreftAdapter,
    DireftUnitAdapter,
    BiLinearAdapter,
)
from reft.dataset import curate_training_data
from reft.intervenable import (
    IntervenableModel,
    IntervenableConfig,
    RepresentationConfig,
)


ADAPTER_CLASS_MAP = {
    "dora": RepresentationDoRA,
    "lora": RepresentationLoRA,
    "add_unit": AdditionUnitIntervention,
    "clamp_unit": ClampUnitIntervention,
    "clamp_free": ClampFreeIntervention,
    "add_free": AdditionFreeIntervention,
    "loreft": LoreftAdapter,
    "direft_unit": DireftUnitAdapter,
    "bilin": BiLinearAdapter,
}


@dataclass
class Arguments:
    seed: int = HfArg(default=42)
    seed_list: List[int] = HfArg(default_factory=list)

    model_path: str = HfArg(default="google/gemma-2-2b-it")
    concept_data_dir: str = HfArg(default="prod_2b_l10_v1/generate")
    output_dir: str = HfArg(default="outputs/")
    layers: List[int] = HfArg(default_factory=list)
    epochs: int = HfArg(default=1)
    batch_size: int = HfArg(default=4)
    learning_rate: float = HfArg(default=1e-3)
    vector_init_scale: float = HfArg(default=1.0)

    factor_learning_rate: float = HfArg(default=10)
    factor_init_scale: float = HfArg(default=4.0)
    factor_learning_rate_list: List[str] = HfArg(default_factory=list)
    factor_init_scale_list: List[str] = HfArg(default_factory=list)

    low_rank_dim: int = HfArg(default=1)
    alpha: float = HfArg(default=1.0)
    adapter_type: str = HfArg(default="dora")
    positions: str = HfArg(default="f4")
    load_in_4bit: bool = HfArg(default=False)
    optimizer: Literal["adam", "sgd"] = HfArg(default="adam")


@torch.no_grad()
def set_decoder_norm_to_unit_norm(lin):
    assert lin.weight is not None, "Decoder weight was not initialized."

    eps = torch.finfo(lin.weight.dtype).eps
    if lin.weight.data.shape[0] > lin.weight.data.shape[1]:
        dim = 0
    else:
        dim = 1
    norm = torch.norm(lin.weight.data, dim=dim, keepdim=True)
    lin.weight.data /= norm + eps


def train_sv(
    args: Arguments,
    model: PreTrainedModel,
    adapter_class,
    train_dataloader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    concept: str,
    concept_id: int,
    do_save: bool = True,
):
    set_seed(args.seed)

    epochs = args.epochs
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
            factor_init_scale=args.factor_init_scale,
            vector_init_scale=args.vector_init_scale,
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
                        "lr": args.learning_rate,
                    }
                )
                param_groups.append(
                    {"params": _lora.learned_source.weight, "lr": args.learning_rate}
                )
                param_groups.append(
                    {"params": _lora.learned_source.bias, "lr": args.learning_rate}
                )
            case "bilin":
                param_groups.append(
                    {"params": _lora.parameters(), "lr": args.learning_rate}
                )
            case _:
                factor_lr = args.factor_learning_rate
                logger.warning(
                    f"Factor LR: {factor_lr:.3f} || Factor init scale: {args.factor_init_scale:.3f}"
                )

                param_groups.append(
                    {"params": _lora.proj.weight, "lr": args.learning_rate}
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
    for epoch_i in pgbar_epoch:
        epoch_loss = 0
        pgbar_step = tqdm(
            train_dataloader, desc=f"Epoch [{epoch_i + 1}/{epochs}]", disable=True
        )
        for batch in pgbar_step:
            locations = batch["intervention_locations"]
            locations = [loc[loc != -1].tolist() for loc in locations]
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
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping should not affect theoretical assumptions.
            # Clip all gradients as a single tensor.
            # This approach is wrong!!!
            # torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            # Only clip vector gradients
            torch.nn.utils.clip_grad_norm_(vector_params, 1.0)

            optimizer.step()
            scheduler.step()

            if args.adapter_type == "add_unit" or args.adapter_type == "clamp_unit":
                for _, _lora in intervenable.interventions.items():
                    set_decoder_norm_to_unit_norm(_lora.proj)

            epoch_loss += loss.item()
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
    args: Arguments, tokenizer: PreTrainedTokenizer, concept_id: int
):
    data_path = Path(args.concept_data_dir, "train_data.parquet")
    df = pd.read_parquet(data_path)

    concept_df = df[df["concept_id"] == concept_id]
    dataset = []
    for _, row in concept_df.iterrows():
        dataset.append((row["input"], row["output"]))

    inputs = []
    for x in dataset:
        ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": x[0]}],
                tokenize=True,
                add_generation_prompt=True,
            )
        if (tokenizer.bos_token is not None) and (ids[0] == tokenizer.bos_token_id):
            ids = ids[1:]
        inputs.append(tokenizer.decode(ids))

    data_module = curate_training_data(
        tokenizer=tokenizer,
        positions=args.positions,
        inputs=inputs,
        outputs=[x[1] for x in dataset],
        padding_side="right",
    )
    train_set, collator = data_module["train_dataset"], data_module["data_collator"]

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataloader = DataLoader(
        train_set,
        collate_fn=collator,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
    )

    logger.warning("**" + tokenizer.decode(train_set[0]["input_ids"]) + "**")
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
        load_in_4bit=args.load_in_4bit,
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

    for seed in args.seed_list:
        for factor_lr in args.factor_learning_rate_list:
            for factor_init in args.factor_init_scale_list:
                logger.warning("============================================================")
                logger.warning(f"Seed: {seed} || Factor lr: {factor_lr} || Factor init: {factor_init}")
                logger.warning("============================================================")

                args.factor_learning_rate = float(factor_lr)
                args.factor_init_scale = float(factor_init)
                args.seed = seed
                args.output_dir = Path(
                    output_base_dir, f"seed={seed}_scale={factor_init}_lr={factor_lr}"
                )

                for concept_id, metarec in metadata.items():
                    save_dir = Path(args.output_dir) / f"{concept_id}"
                    if (save_dir / "state_dict.pt").exists():
                        logger.warning(f"Already done concept id: {concept_id}; skipping")
                        continue

                    train_dataloader = get_concept_dataloader(
                        args, tokenizer, concept_id
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
                    )


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    main(parser.parse_args())
