from collections import OrderedDict
from dataclasses import dataclass
import json
import logging
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
)
from transformers.hf_argparser import HfArg

from reft.interventions import ParameterDoRA, ParameterLoRA
from reft.utils import curate_data, load_hf_model_tokenizer, logger_setup


logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    seed: int = HfArg(default=42)
    model_path: str = HfArg(default="google/gemma-2-2b-it")
    output_dir: str = HfArg(default="outputs/")
    layers: List[int] = HfArg(default_factory=list)
    epochs: int = HfArg(default=1)
    batch_size: int = HfArg(default=4)
    learning_rate: float = HfArg(default=1e-3)
    concept_id: int = HfArg(default=0)
    low_rank_dim: int = HfArg(default=1)
    alpha: float = HfArg(default=1.0)
    adapter_type: Literal["lora", "dora"] = HfArg(default="dora")
    load_in_4bit: bool = HfArg(default=False)


def main(args: Arguments):
    set_seed(args.seed)
    logger_setup(logger=logger)
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
        padding_side='left',
        load_in_4bit=args.load_in_4bit,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    train_set, collator = curate_data(
        model=model,
        tokenizer=tokenizer,
        inputs=[
            tokenizer.decode(tokenizer.apply_chat_template(
                [{"role": "user", "content": x[0]}],
                tokenize=True,
                add_generation_prompt=True,
            )[1:])
            for x in dataset
        ],
        outputs=[x[1] for x in dataset],
        eos_token="<end_of_turn>",
    )
    train_dataloader = DataLoader(
        train_set, collate_fn=collator, batch_size=args.batch_size, shuffle=True
    )

    logger.warning(tokenizer.decode(train_set[0]['input_ids']))

    epochs = args.epochs
    embed_dim = model.config.hidden_size
    alpha = args.alpha

    if args.adapter_type == "dora":
        adapter_class = ParameterDoRA
    elif args.adapter_type == "lora":
        adapter_class = ParameterLoRA
    else:
        raise ValueError(f"Unknown adapter type: {args.adapter_type}")

    dora = OrderedDict()
    target_modules = []
    for layer_i in args.layers:
        module_name = f"model.layers.{layer_i}.self_attn.o_proj"
        target_lin: nn.Linear = model.get_submodule(module_name)
        _dora = adapter_class(
            lin=target_lin,
            in_dim=target_lin.in_features,
            out_dim=target_lin.out_features,
            low_rank_dim=args.low_rank_dim,
            alpha=args.alpha,
        ).to(device=device, dtype=dtype)
        model.set_submodule(module_name, _dora)
        dora[module_name] = _dora
        target_modules.append(module_name)

    trainable_params = []
    for _, _dora in dora.items():
        trainable_params.extend(list(_dora.parameters()))
    num_trainable_parameters = sum(p.numel() for p in trainable_params)

    logger.warning(
        f"Trainable parameters: {num_trainable_parameters} || {num_trainable_parameters / sum(p.numel() for p in model.parameters())*100:.6f}%"
    )
    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_training_steps=epochs * len(train_dataloader),
        num_warmup_steps=0,
    )

    pgbar = tqdm(range(epochs))
    for epoch_i in pgbar:
        epoch_loss = 0
        for batch in train_dataloader:
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            logits = outputs.logits[:, :-1].contiguous()
            shift_logits = logits.view(-1, logits.size(-1))
            labels = batch['labels'][:, 1:].contiguous().to(device)
            shift_labels = labels.view(-1)
            loss = nn.functional.cross_entropy(shift_logits, shift_labels, reduction='mean')
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()

            epoch_loss += loss.item()
        epoch_loss /= len(train_dataloader)
        pgbar.set_postfix({'loss': f"{epoch_loss:.6f}"})
        torch.cuda.empty_cache()
    pgbar.close()

    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    state_dict = OrderedDict()
    for mod in target_modules:
        _state_dict = OrderedDict(
            [(k, v.data.cpu()) for k, v in dora[mod].state_dict().items()]
        )
        state_dict[mod] = _state_dict
    save_path = save_dir / "state_dict.pt"
    torch.save(state_dict, save_path)
    logger.warning(f"Saved to `{save_path}`")

    cfg = {
        "embed_dim": embed_dim,
        "low_rank_dim": args.low_rank_dim,
        "layers": args.layers,
        "target_modules": target_modules,
        "alpha": alpha,
        "concept": concept,
        "class": adapter_class.__name__,
    }
    save_path = save_dir / "config.json"
    with open(save_path, 'w') as fp:
        json.dump(cfg, fp, indent=2)
    logger.warning(f"Saved to `{save_path}`")


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    main(parser.parse_args())
