from dataclasses import dataclass
import httpx
import json
from loguru import logger
from openai import AsyncClient
import os
import pandas as pd
from pathlib import Path
from typing import List

from transformers import (
    HfArgumentParser,
)
from transformers.hf_argparser import HfArg

from reft.llm_judge import judge_async


@dataclass
class Arguments:
    output_dir: str = HfArg(default="outputs/")
    concept_data_dir: str = HfArg(default="prod_2b_l10_v1/generate")
    batch_size: int = HfArg(default=4)
    num_examples: int = HfArg(default=None)
    positions: str = HfArg(default="f4")
    judge_lm: str = HfArg(default="gpt-4o-mini")
    max_concepts: int = HfArg(default=None)


def eval_concept_df(
    args: Arguments,
    client: AsyncClient,
    steer_df: pd.DataFrame,
    batch_size: int,
    concept: str,
    concept_id: int,
):
    logger.warning(f"[Concept {concept_id}]: {concept}")

    concepts = []
    instructions = []
    responses = []
    for _, row in steer_df.iterrows():
        prompt = row["original_prompt"]
        output = row["steered_generation"]

        concepts.append(concept)
        instructions.append(prompt)
        responses.append(output)

    eval_df = judge_async(
        client=client,
        concept_id=concept_id,
        concepts=concepts,
        instructions=instructions,
        responses=responses,
        batch_size=batch_size,
        judge_lm=args.judge_lm,
        max_new_tokens=500,
    )
    eval_df["concept"] = [concept] * len(concepts)

    return pd.DataFrame(eval_df)


def main(args: Arguments):
    logger.warning(args)

    metadata_path = Path(args.concept_data_dir, 'metadata.jsonl')
    metadata = {}
    for line in open(metadata_path).readlines():
        rec = json.loads(line)
        metadata[rec['concept_id']] = rec
    if args.max_concepts is not None:
        logger.warning(f"Using first {args.max_concepts} concepts")
        metadata = {k: v for k, v in metadata.items() if k < args.max_concepts}
    logger.warning(f"Number of concepts: {len(metadata.keys())}")

    client = AsyncClient(
        api_key=os.environ.get("OPENAI_API_KEY", None),
        base_url=os.environ.get("OPENAI_BASE_URL", None),
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=1000),
            headers={"Connection": "close"},
        ),
        max_retries=10,
        timeout=60.0,
    )

    output_base_dir = args.output_dir

    for concept_id, metarec in metadata.items():
        concept = metarec['concept']
        dump_dir = Path(output_base_dir, f"{concept_id}")
        save_path = dump_dir / "eval.parquet"

        if save_path.exists():
            logger.warning(f"Already done concept id {concept_id}; skipping")
            continue
        
        steered_generations_path = dump_dir / "suppression_steered_generations.parquet"
        if not steered_generations_path.exists():
            logger.warning(f"No generations for concept id: {concept_id}; skipping")
            continue
        
        steer_df = pd.read_parquet(steered_generations_path)

        eval_df = eval_concept_df(
            args=args,
            client=client,
            steer_df=steer_df,
            batch_size=args.batch_size,
            concept=concept,
            concept_id=concept_id,
        )
        if save_path.exists():
            logger.warning(f"Overwriting {save_path}")
            old_df = pd.read_parquet(save_path)
            eval_df = pd.concat(
                [old_df, eval_df], axis=0, ignore_index=True
            )
        eval_df.to_parquet(save_path, index=False)
        logger.warning(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    main(parser.parse_args())
