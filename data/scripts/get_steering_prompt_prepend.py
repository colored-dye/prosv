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

from reft.llm_judge import steering_prompt_prepend_async


@dataclass
class Arguments:
    concept_data_dir: str = HfArg(default="prod_2b_l10_v1/generate")
    batch_size: int = HfArg(default=4)
    max_concepts: int=HfArg(default=None)
    judge_lm: str = HfArg(default="gpt-4o-mini")


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

    dump_dir = Path(args.concept_data_dir)
    save_path = dump_dir / "steering_prompt_prepend.parquet"

    concept_ids = []
    concepts = []
    for concept_id, metarec in metadata.items():
        concept = metarec['concept']
        concept_ids.append(concept_id)
        concepts.append(concept)

    if save_path.exists():
        old_df = pd.read_parquet(save_path)
        done_concept_ids = old_df["concept_id"].unique()
        done_concepts = old_df["concept"].unique()
        logger.warning(f"Already done {len(done_concepts)}")

        for c in done_concept_ids:
            if c in concept_ids:
                concept_ids.remove(c)
        for c in done_concepts:
            if c in concepts:
                concepts.remove(c)

    steer_df = steering_prompt_prepend_async(
        client=client,
        concepts=concepts,
        batch_size=args.batch_size,
        judge_lm=args.judge_lm,
        max_new_tokens=500,
    )
    steer_df["concept_id"] = concept_ids
    steer_df = pd.DataFrame(steer_df)

    if save_path.exists():
        logger.warning(f"Appending to {save_path}")
        old_df = pd.read_parquet(save_path)
        steer_df = pd.concat([old_df, steer_df], axis=0, ignore_index=True)
    steer_df.to_parquet(save_path, index=False)
    logger.warning(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    main(parser.parse_args())
