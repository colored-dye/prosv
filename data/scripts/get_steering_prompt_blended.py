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

from reft.llm_judge import steering_prompt_blended_async


@dataclass
class Arguments:
    concept_data_dir: str = HfArg(default="prod_2b_l10_v1/generate")
    alpaca_eval_path: str = HfArg(default="alpaca_eval.json")
    batch_size: int = HfArg(default=4)
    num_examples: int = HfArg(default=None)
    judge_lm: str = HfArg(default="gpt-4o-mini")
    num_instructions_per_concept: int = HfArg(default=10)


def generate_steering_prompt(
    args: Arguments,
    client: AsyncClient,
    instructions: List[str],
    batch_size: int,
    concept: str,
    concept_id: int,
):
    logger.warning(f"[Concept {concept_id}]: {concept}")

    steer_df = steering_prompt_blended_async(
        client=client,
        concept_id=concept_id,
        concepts=[concept]*len(instructions),
        instructions=instructions,
        batch_size=batch_size,
        judge_lm=args.judge_lm,
        max_new_tokens=500,
    )
    steer_df["concept"] = [concept] * len(instructions)

    return pd.DataFrame(steer_df)


def main(args: Arguments):
    logger.warning(args)

    metadata_path = Path(args.concept_data_dir, 'metadata.jsonl')
    metadata = {}
    for line in open(metadata_path).readlines():
        rec = json.loads(line)
        metadata[rec['concept_id']] = rec

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

    alpaca_eval_dataset = pd.read_json(args.alpaca_eval_path)

    for concept_id, metarec in metadata.items():
        concept = metarec['concept']
        dump_dir = Path(args.concept_data_dir)
        save_path = dump_dir / "steering_prompt_blended.parquet"

        sampled_prompts = alpaca_eval_dataset.sample(
            args.num_instructions_per_concept, random_state=concept_id
        )["instruction"].tolist()

        if save_path.exists():
            old_df = pd.read_parquet(save_path)
            done_concepts = old_df["concept_id"].unique()
            if concept_id in done_concepts:
                logger.warning(f"Already done concept id {concept_id}; skipping")
                continue

        steer_df = generate_steering_prompt(
            args=args,
            client=client,
            instructions=sampled_prompts,
            batch_size=args.batch_size,
            concept=concept,
            concept_id=concept_id,
        )
        if save_path.exists():
            logger.warning(f"Appending to {save_path}")
            old_df = pd.read_parquet(save_path)
            steer_df = pd.concat([old_df, steer_df], axis=0, ignore_index=True)
        steer_df.to_parquet(save_path, index=False)
        logger.warning(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    main(parser.parse_args())
