from dataclasses import dataclass
import json
from loguru import logger
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import (
    set_seed,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.hf_argparser import HfArg

from reft.dataset import curate_training_data, curate_vanilla_training_data
from reft.utils import (
    load_hf_model_tokenizer,
)
from reft.intervenable import IntervenableModel, InterventionMode


@dataclass
class Arguments:
    seed: int = HfArg(default=42)

    model_path: str = HfArg(default="google/gemma-2-2b-it")
    output_dir: str = HfArg(default="outputs/")
    concept_data_dir: str = HfArg(default="prod_2b_l10_v1/generate")
    alpaca_eval_path: str = HfArg(default="alpaca_eval.json")
    batch_size: int = HfArg(default=4)
    max_new_tokens: int = HfArg(default=128)
    load_in_4bit: bool = HfArg(default=False)
    num_examples: int = HfArg(default=None)
    num_instructions_per_concept: int = HfArg(default=10)
    positions: str = HfArg(default="f4")
    prefix_length: int = HfArg(default=None)
    max_concepts: int = HfArg(default=None)

    batch_size: int = HfArg(default=None)


def steer_generation(
    args: Arguments,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    load_dir: Path,
    test_dataloader: DataLoader,
    concept: str,
    concept_id: int,
):
    set_seed(args.seed)

    if args.positions != "prompt":
        intervenable = IntervenableModel.load(load_directory=load_dir, model=model)
    logger.warning(f"[Concept {concept_id}]: {concept}")

    all_responses = []
    for batch in tqdm(test_dataloader, desc=f"Steering on concept id [{concept_id}]"):
        if args.positions != "prompt": # SV steering
            locations = batch["intervention_locations"]
            if args.positions == "all": # Special treatment for full-sequence generation
                intervention_mode = InterventionMode.ALL_GENERATION
            else:
                intervention_mode = InterventionMode.PROMPT_ONLY

            outputs = intervenable.generate(
                locations=locations,
                intervention_mode=intervention_mode,
                input_ids=batch["input_ids"].to(intervenable.device),
                attention_mask=batch["attention_mask"].to(intervenable.device),
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                use_cache=True,
            )
        else: # Prompt steering
            outputs = model.generate(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                use_cache=True,
            )

        out_text = tokenizer.batch_decode(
            outputs[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
        )
        all_responses.extend(out_text)
    return all_responses


def get_test_data(args: Arguments, tokenizer: PreTrainedTokenizer, concept_id: int):
    if args.positions == "prompt": # Prompt steering
        # Blended steering prompt
        # df = pd.read_parquet(Path(args.concept_data_dir, "steering_prompt_blended.parquet"))
        # df = df[df['concept_id']==concept_id]
        # inputs = []
        # for _, row in df.iterrows():
        #     x = row['steering_prompt']
        #     ids = tokenizer.apply_chat_template(
        #             [{"role": "user", "content": x}],
        #             tokenize=True,
        #             add_generation_prompt=True,
        #         )
        #     if (tokenizer.bos_token is not None) and (ids[0] == tokenizer.bos_token_id):
        #         ids = ids[1:]
        #     inputs.append(tokenizer.decode(ids))
        # sampled_prompts = df['original_prompt'].tolist()

        # Prepend steering prompt
        df = pd.read_parquet(Path(args.concept_data_dir, "steering_prompt_prepend.parquet"))
        df = df[df['concept_id']==concept_id]
        steering_prompt = df['steering_prompt'].iloc[0]

        inputs = []
        alpaca_eval_dataset = pd.read_json(args.alpaca_eval_path)
        sampled_prompts = alpaca_eval_dataset.sample(
            args.num_instructions_per_concept, random_state=concept_id
        )["instruction"].tolist()

        for x in sampled_prompts:
            steered_prompt = f"{steering_prompt}\n\nQuestion: {x}"
            ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": steered_prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            if (tokenizer.bos_token is not None) and (ids[0] == tokenizer.bos_token_id):
                ids = ids[1:]
            inputs.append(tokenizer.decode(ids))

        data_module = curate_vanilla_training_data(
            tokenizer=tokenizer,
            inputs=inputs,
            outputs=["" for _ in inputs],
            padding_side="left",
        )
    else: # SV steering
        alpaca_eval_dataset = pd.read_json(args.alpaca_eval_path)
        sampled_prompts = alpaca_eval_dataset.sample(
            args.num_instructions_per_concept, random_state=concept_id
        )["instruction"].tolist()

        inputs = []
        for x in sampled_prompts:
            ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": x}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            if (tokenizer.bos_token is not None) and (ids[0] == tokenizer.bos_token_id):
                ids = ids[1:]
            inputs.append(tokenizer.decode(ids))

        if args.prefix_length is not None:
            logger.critical(f"Intervention location skips prefix {args.prefix_length}")

        data_module = curate_training_data(
            tokenizer=tokenizer,
            positions=args.positions,
            inputs=inputs,
            outputs=["" for _ in inputs],
            padding_side="left",
            prefix_length=args.prefix_length,
        )

    train_set, collator = data_module["train_dataset"], data_module["data_collator"]
    train_dataloader = DataLoader(
        train_set, collate_fn=collator, batch_size=args.batch_size, shuffle=False
    )

    logger.warning("**" + tokenizer.decode(train_set[0]["input_ids"]) + "**")
    return sampled_prompts, train_dataloader


def main(args: Arguments):
    logger.warning(args)
    set_seed(args.seed)

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
        concept = metarec["concept"]
        dump_dir = Path(args.output_dir, f"{concept_id}")
        dump_dir.mkdir(parents=True, exist_ok=True)
        save_path = dump_dir / "steered_generations.parquet"

        if save_path.exists():
            logger.warning(
                f"Already done concept id: {concept_id}; skipping"
            )
            continue

        if args.positions != "prompt" and not (dump_dir / "state_dict.pt").exists():
            logger.warning(
                f"No checkpoint for concept id: {concept_id}; skipping"
            )
            continue

        test_prompts, test_dataloader = get_test_data(
            args, tokenizer, concept_id
        )
        generations = steer_generation(
            args=args,
            model=model,
            tokenizer=tokenizer,
            load_dir=dump_dir,
            test_dataloader=test_dataloader,
            concept_id=concept_id,
            concept=concept,
        )
        save_df = {
            "original_prompt": test_prompts,
            "steered_generation": generations,
            "concept": [concept] * len(test_prompts),
            "concept_id": [concept_id] * len(test_prompts),
        }
        save_df = pd.DataFrame(save_df)

        # if save_path.exists():
        #     logger.warning(f"Overwriting {save_path}")
        #     old_df = pd.read_parquet(save_path)
        #     save_df = pd.concat([old_df, save_df], axis=0, ignore_index=True)
        save_df.to_parquet(save_path, index=False)
        logger.warning(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    main(parser.parse_args())
