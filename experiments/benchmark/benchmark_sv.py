from dataclasses import dataclass
import json
from loguru import logger
import pandas as pd
from pathlib import Path
from typing import List, Literal

import torch
from transformers import (
    HfArgumentParser,
)
from transformers.hf_argparser import HfArg

from reft.utils import (
    load_hf_model_tokenizer,
)
from reft.intervenable import IntervenableModel

from benchmark_utils import mmlu_eval, tinymmlu_eval, tinygsm8k_eval, mbpp_eval


BENCHMARK_FUNCTION_MAPPING = {
    "mmlu": mmlu_eval,
    "tinymmlu": tinymmlu_eval,
    "tinygsm8k": tinygsm8k_eval,
    "mbpp": mbpp_eval,
}


@dataclass
class Arguments:
    seed: int = HfArg(default=42)

    model_path: str = HfArg(default="google/gemma-2-2b-it")
    ckpt_dir: str = HfArg(default="outputs/")
    save_dir: str = HfArg(default="outputs/")
    concept_data_dir: str = HfArg(default="prod_2b_l10_v1/generate")
    batch_size: int = HfArg(default=4)
    max_new_tokens: int = HfArg(default=128)
    load_in_4bit: bool = HfArg(default=False)
    num_examples: int = HfArg(default=None)
    positions: str = HfArg(default="f4")
    num_shots: int = HfArg(default=0)
    benchmark_name: Literal[
        "mmlu",
        "tinymmlu",
        "tinygsm8k",
        "mbpp",
    ] = HfArg(default="mmlu")
    benchmark_hf_path: str = HfArg(default="cais/mmlu")
    no_interventions: bool = HfArg(default=False)
    prompt_steering: bool = HfArg(default=False)
    max_concepts: int = HfArg(default=None)


def main(args: Arguments):
    logger.warning(args)

    device = "cuda"
    dtype = torch.bfloat16
    model, tokenizer = load_hf_model_tokenizer(
        model_name_or_path=args.model_path,
        device=device,
        dtype=dtype,
        padding_side="left",
        load_in_4bit=args.load_in_4bit,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    metadata_path = Path(args.concept_data_dir, 'metadata.jsonl')
    metadata = {}
    for line in open(metadata_path).readlines():
        rec = json.loads(line)
        metadata[rec['concept_id']] = rec
    if args.max_concepts is not None:
        logger.warning(f"Using first {args.max_concepts} concepts")
        metadata = {k: v for k, v in metadata.items() if k < args.max_concepts}
    logger.warning(f"Number of concepts: {len(metadata.keys())}")

    for concept_id, metarec in metadata.items():
        concept = metarec['concept']
        logger.warning(f"[Concept {concept_id}]: `{concept}`")

        dump_dir = Path(args.ckpt_dir, f"{concept_id}")
        save_dir = Path(args.save_dir) / f"{concept_id}"
        save_path = save_dir / f"steered_benchmark_{args.benchmark_name}.parquet"

        if args.no_interventions or args.prompt_steering:
            intervenable = model
        else:
            if save_path.exists():
                logger.warning(f"Already done concept id: {concept_id}; skipping")
                continue

            if (not args.no_interventions or not args.prompt_steering) and not (dump_dir / "state_dict.pt").exists():
                logger.warning(f"No checkpoint for concept id: {concept_id}; skipping")
                continue

            intervenable = IntervenableModel.load(load_directory=dump_dir, model=model)

            # TODO: remove this since this is ablation
            for k, intv in intervenable.interventions.items():
                intv.factor.data *= 0.5

        steering_prompt = None
        if args.prompt_steering:
            logger.warning("Loading steering prompts")
            steering_prompt_df = pd.read_parquet(
                Path(
                    args.concept_data_dir, "steering_prompt_prepend.parquet"
                )
            )
            steering_prompt = steering_prompt_df[
                steering_prompt_df["concept_id"] == concept_id
            ].iloc[0]['steering_prompt']
            logger.warning(f"Steering prompt: **{steering_prompt}**")

        benchmark_eval_fn = BENCHMARK_FUNCTION_MAPPING[args.benchmark_name]
        benchmark_results = benchmark_eval_fn(
            intervenable=intervenable,
            tokenizer=tokenizer,
            positions=args.positions,
            batch_size=args.batch_size,
            seed=args.seed,
            hf_dataset_path=args.benchmark_hf_path,
            num_shots=args.num_shots,
            max_new_tokens=args.max_new_tokens,
            logger=logger,
            no_interventions=args.no_interventions,
            steering_prompt=steering_prompt,
        )

        save_df = pd.DataFrame(benchmark_results)

        if save_path.exists():
            logger.critical(f"Overwriting {save_path}")

            # logger.warning(f"Appending to {save_path}")
            # old_df = pd.read_parquet(save_path)
            # save_df = pd.concat([old_df, save_df], axis=0, ignore_index=True)

        save_dir.mkdir(parents=True, exist_ok=True)
        save_df.to_parquet(save_path, index=False)
        logger.warning(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    main(parser.parse_args())
