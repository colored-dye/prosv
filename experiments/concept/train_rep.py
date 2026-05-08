from collections import OrderedDict
from dataclasses import dataclass
import json
from loguru import logger
import math
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
)
from transformers.hf_argparser import HfArg

from reft.nethook import TraceDict
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
from reft.intervenable import _spawn_edit_fn


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
    model_path: str = HfArg(default="google/gemma-2-2b-it")
    output_dir: str = HfArg(default="outputs/")
    layers: List[int] = HfArg(default_factory=list)
    epochs: int = HfArg(default=1)
    batch_size: int = HfArg(default=4)
    learning_rate: float = HfArg(default=1e-3)
    factor_learning_rate: float = HfArg(default=10)
    concept_id: int = HfArg(default=0)
    low_rank_dim: int = HfArg(default=1)
    alpha: float = HfArg(default=1.0)
    adapter_type: str = HfArg(default="dora")
    factor_init_scale: float = HfArg(default=1.0)
    vector_init_scale: float = HfArg(default=1.0)
    positions: str = HfArg(default="f4")
    load_in_4bit: bool = HfArg(default=False)
    optimizer: Literal["adam", "sgd"] = HfArg(default="adam")


@torch.no_grad()
def set_decoder_norm_to_unit_norm(lin):
    assert lin.weight is not None, "Decoder weight was not initialized."

    eps = torch.finfo(lin.weight.dtype).eps
    if lin.weight.data.shape[0] > lin.weight.data.shape[1]:
        dim=0
    else:
        dim=1
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
    do_save: bool = True,
):
    epochs = args.epochs
    embed_dim = model.config.hidden_size
    alpha = args.alpha

    target_modules = []
    lora = OrderedDict()
    for layer_i in args.layers:
        module_name = f"model.layers.{layer_i}"
        target_modules.append(module_name)
        _lora_device = tuple(model.get_submodule(module_name).parameters())[0].device
        _lora = adapter_class(
            embed_dim=embed_dim,
            low_rank_dim=args.low_rank_dim,
            alpha=alpha,
            factor_init_scale=args.factor_init_scale,
            vector_init_scale=args.vector_init_scale,
        ).to(device=_lora_device, dtype=dtype)
        lora[module_name] = _lora

    trainable_params = []
    param_groups = []
    for _, _lora in lora.items():
        trainable_params.extend(list(_lora.parameters()))

        match args.adapter_type:
            case "loreft" | "direft_unit":
                param_groups.append({"params": _lora.rotate_layer.parameters(), "lr": args.learning_rate})
                param_groups.append({"params": _lora.learned_source.weight, "lr": args.learning_rate})
                param_groups.append({"params": _lora.learned_source.bias, "lr": args.learning_rate})
            case "bilin":
                param_groups.append({"params": _lora.parameters(), "lr": args.learning_rate})
            case _:
                # factor_lr = -10*math.log10(args.learning_rate)
                # factor_lr = 1e-2 / args.learning_rate
                # factor_lr = args.learning_rate
                factor_lr = args.factor_learning_rate
                logger.warning(f"Factor LR: {factor_lr:.3f} || Factor init scale: {args.factor_init_scale:.3f}")

                param_groups.append({"params": _lora.proj.weight, "lr": args.learning_rate})
                param_groups.append({"params": _lora.factor, "lr": factor_lr})

    num_trainable_params = sum(p.numel() for p in trainable_params if p.requires_grad)
    num_model_params = sum(p.numel() for p in model.parameters())
    ratio_trainable_params = num_trainable_params / num_model_params
    logger.warning(
        f"Trainable parameters: {num_trainable_params} || {ratio_trainable_params*100:.3e}%"
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

    all_step_loss = []
    all_epoch_loss = []
    all_step_factor = []
    pgbar = tqdm(range(epochs), desc="Epochs")
    for epoch_i in pgbar:
        epoch_loss = 0
        pgbar_step = tqdm(
            train_dataloader, desc=f"Epoch [{epoch_i + 1}/{epochs}]", disable=True
        )
        for batch in pgbar_step:
            locations = batch["intervention_locations"]
            locations = [loc[loc!=-1].tolist() for loc in locations]
            edit_fn = _spawn_edit_fn(
                interventions=lora,
                locations=locations
            )

            with TraceDict(
                module=model,
                layers=target_modules,
                retain_output=False,
                retain_input=False,
                retain_grad=False,
                edit_output=edit_fn,
            ):
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
            logits = outputs.logits[:, :-1].contiguous()
            shift_logits = logits.view(-1, logits.size(-1))
            labels = batch['labels'][:, 1:].contiguous().to(device)
            shift_labels = labels.view(-1)
            loss = nn.functional.cross_entropy(
                shift_logits, shift_labels, reduction="mean"
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            if args.adapter_type == "add_unit" or args.adapter_type == "clamp_unit":
                for _, _lora in lora.items():
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
            torch.cuda.empty_cache()

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
        pgbar.set_postfix(pgbar_log)
        torch.cuda.empty_cache()
    pgbar.close()

    # results_path = Path('results/9b_l20.csv')
    # if results_path.exists():
    #     results_df = pd.read_csv(results_path)
    # new_res = {}
    # new_res['method'] = args.adapter_type
    # new_res['factor_init_scale'] = args.factor_init_scale
    # new_res['factor_lr'] = args.factor_learning_rate
    # new_res['train_loss'] = epoch_loss
    # if results_path.exists():
    #     results_df = pd.concat([results_df, pd.DataFrame([new_res])], ignore_index=True)
    # else:
    #     results_df = pd.DataFrame([new_res])
    # results_df.to_csv(results_path, index=False)

    if do_save:
        save_dir = Path(args.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
        state_dict = OrderedDict()
        for layer_i, module_name in zip(args.layers, target_modules):
            _state_dict = OrderedDict([(k, v.data.cpu()) for k, v in lora[module_name].state_dict().items()])
            state_dict[module_name] = _state_dict
        save_path = save_dir / "state_dict.pt"
        torch.save(state_dict, save_path)
        logger.warning(f"Saved to `{save_path}`")
    
        log = {
            "losses_epoch": all_epoch_loss,
            "losses_step": all_step_loss,
            "factor_step": all_step_factor,
        }
        save_path = save_dir / "log.pt"
        torch.save(log, save_path)
        logger.warning(f"Saved to `{save_path}`")
    
        cfg = {
            "embed_dim": embed_dim,
            "low_rank_dim": args.low_rank_dim,
            # "alpha": alpha,
            "concept": concept,
            "target_modules": target_modules,
            "layers": args.layers,
            "class": adapter_class.__name__,
        }
        save_path = save_dir / "config.json"
        with open(save_path, 'w') as fp:
            json.dump(cfg, fp, indent=2)
        logger.warning(f"Saved to `{save_path}`")


def main(args: Arguments):
    set_seed(args.seed)
    logger.warning(args)

    df = pd.read_parquet('train_data.parquet')
    concept_df = df[df['concept_id']==args.concept_id]
    concept = concept_df.iloc[0]['output_concept']
    logger.warning(f"Concept: `{concept}`")
    dataset = []
    for _, row in concept_df.iterrows():
        rec = []
        rec.append(row['input'])
        rec.append(row['output'])
        dataset.append(rec)

    device = 'cuda'
    dtype = torch.bfloat16
    model, tokenizer = load_hf_model_tokenizer(
        model_name_or_path=args.model_path,
        device=device,
        dtype=dtype,
        padding_side="right",
        load_in_4bit=args.load_in_4bit,
    )

    data_module = curate_training_data(
        tokenizer=tokenizer,
        positions=args.positions,
        inputs=[
            tokenizer.decode(tokenizer.apply_chat_template(
                [{"role": "user", "content": x[0]}],
                tokenize=True,
                add_generation_prompt=True,
            )[1:])
            for x in dataset
        ],
        outputs=[x[1] for x in dataset],
        padding_side="right",
    )
    train_set, collator = data_module["train_dataset"], data_module["data_collator"]
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataloader = DataLoader(
        train_set, collate_fn=collator, batch_size=args.batch_size, shuffle=True, generator=g,
    )

    logger.warning("**" + tokenizer.decode(train_set[0]["input_ids"]) + "**")

    adapter_class = ADAPTER_CLASS_MAP.get(args.adapter_type)
    if adapter_class is None:
        raise ValueError(f"Unknown adapter type: `{args.adapter_type}`")

    train_sv(
        args=args,
        model=model,
        adapter_class=adapter_class,
        train_dataloader=train_dataloader,
        device=device,
        dtype=dtype,
        concept=concept,
    )

    # # factor_lrs = [0.5, 1.0, 4.0, 8.0, 12.0, 16.0, 25.0, 32.0]
    # factor_lrs = [0.01, 0.05, 0.1, 0.5, 1.0, 4.0, 8.0, 12.0, 16.0, 25.0, 32.0]
    # # factor_init_scales = [1.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    # factor_init_scales = [1.0]
    # for factor_lr in factor_lrs:
    #     for factor_init_scale in factor_init_scales:
    #         args.factor_learning_rate = factor_lr
    #         args.factor_init_scale = factor_init_scale
    #         train_sv(
    #             args=args,
    #             model=model,
    #             adapter_class=adapter_class,
    #             train_dataloader=train_dataloader,
    #             device=device,
    #             dtype=dtype,
    #             concept=concept,
    #         )


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    main(parser.parse_args())
