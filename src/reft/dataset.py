"""
This module is based on and inspired by:
https://github.com/stanfordnlp/pyreft/blob/main/pyreft/dataset.py.
"""

import copy
from dataclasses import dataclass
from datasets import Dataset
from loguru import logger
import torch
from transformers import PreTrainedTokenizer, DefaultDataCollator
from tqdm import tqdm
from typing import Any, Dict, List, Literal

from .constants import IGNORE_INDEX, LOCATION_PAD


def parse_positions(positions: str):
    """
    Parse prefix/suffix positions.

    Args:
        positions: f2, l3, f4+l5.
    """
    first_n, last_n = 0, 0
    if "+" in positions:
        first_n = int(positions.split("+")[0].strip("f"))
        last_n = int(positions.split("+")[1].strip("l"))
    else:
        if "f" in positions:
            first_n = int(positions.strip("f"))
        elif "l" in positions:
            last_n = int(positions.strip("l"))
    return first_n, last_n


def get_intervention_locations(positions, last_position) -> List[List[int]]:
    """
    Return a 2D list of indices as intervention locations.

    Args:
        positions: f2, l3, f4+l5, all.
        last_position: Index **after** the last intervention.
    """
    assert last_position > 0

    if positions == "all":
        return [list(range(last_position))]
    else:
        first_n, last_n = parse_positions(positions)
        first_n = min(first_n, last_position)
        last_n = min(last_n, last_position)
        if first_n + last_n > last_position:
            last_n = last_position - first_n

        left_positions = list(range(first_n))
        right_positions = list(range(last_position - last_n, last_position))
        return [left_positions + right_positions]


@dataclass
class ReftDataCollator:
    """
    Args:
        padding_side: Use right padding for training and left padding for inference.
    """

    data_collator: DefaultDataCollator
    tokenizer: PreTrainedTokenizer
    padding_side: Literal["left", "right"] = "right"

    def __call__(self, instances: List[Dict[str, Any]]):
        max_seq_len = -1
        for inst in instances:
            max_seq_len = max(max_seq_len, len(inst["input_ids"]))

        for inst in instances:
            non_pad_len = len(inst["input_ids"])

            intervention_location_paddings = torch.tensor(
                [
                    LOCATION_PAD
                    for _ in range(max_seq_len - len(inst["intervention_locations"]))
                ]
            )
            if self.padding_side == "right":
                inst["intervention_locations"] = torch.cat(
                    [inst["intervention_locations"], intervention_location_paddings]
                ).long()
            else:
                inst["intervention_locations"] = torch.cat(
                    [
                        intervention_location_paddings,
                        inst["intervention_locations"] + max_seq_len - non_pad_len,
                    ]
                ).long()

            input_ids_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - non_pad_len)]
            )
            if self.padding_side == "right":
                inst["input_ids"] = torch.cat(
                    [inst["input_ids"], input_ids_paddings]
                ).int()
            else:
                inst["input_ids"] = torch.cat(
                    [input_ids_paddings, inst["input_ids"]]
                ).int()

            labels_paddings = torch.tensor(
                [IGNORE_INDEX for _ in range(max_seq_len - non_pad_len)]
            )
            if self.padding_side == "right":
                inst["labels"] = torch.cat([inst["labels"], labels_paddings]).long()
            else:
                inst["labels"] = torch.cat([labels_paddings, inst["labels"]]).long()

            inst["attention_mask"] = (
                inst["input_ids"] != self.tokenizer.pad_token_id
            ).int()

        batch = self.data_collator(instances)
        return batch


def curate_training_data(
    tokenizer: PreTrainedTokenizer,
    inputs: list,
    outputs: list,
    positions: str = "f4",
    padding_side: Literal["right", "left"] = "right",
    prompt_max_length: int = 4096,
    max_length: int = 4096,
    eos_token: str = None,
    prefix_length: int = None,
):
    """
    Data curation for both training and inference,
    with intervention locations.

    Args:
        positions: f2, l3, f4+l5, all_prompt, all.
            * f2: First 2 prompt tokens.
            * l3: Last 3 prompt tokens.
            * f4+l5: First 4 and last 5 prompt tokens.
            * all_prompt: All prompt tokens.
            * all: All or prompt + response.
        padding_side: "right" for training and "left" for inference.
            Right padding is essential for intervention training,
            since we do not want interventions on
            padding tokens to affect actual tokens.
        eos_token: Not append EOS token if None.

    Returns:
        {"train_dataset": ..., "data_collator": ...}
    """
    all_base_input_ids = []
    all_output_ids = []
    all_intervention_locations = []

    disable_tqdm = False if len(inputs) > 1000 else True # Silence if dataset is small
    for i in tqdm(range(len(inputs)), desc="Preprocessing data", disable=disable_tqdm):
        _input = inputs[i]
        _output = outputs[i]

        base_prompt = _input
        base_input = base_prompt + _output
        if eos_token is not None:
            base_input += eos_token

        base_prompt_ids = tokenizer(
            base_prompt,
            max_length=prompt_max_length,
            truncation=True,
        )["input_ids"]
        base_input_ids = tokenizer(
            base_input,
            max_length=max_length,
            truncation=True,
        )["input_ids"]

        if tokenizer.bos_token is not None:
            # Prepend bos token if chat template needs one; remove otherwise
            if (
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": "foo"}], tokenize=True
                )[0]
                == tokenizer.bos_token_id
            ):
                if base_prompt_ids[0] != tokenizer.bos_token_id:
                    base_prompt_ids = [tokenizer.bos_token_id] + base_prompt_ids
                if base_input_ids[0] != tokenizer.bos_token_id:
                    base_input_ids = [tokenizer.bos_token_id] + base_input_ids
            else:
                if base_prompt_ids[0] == tokenizer.bos_token_id:
                    base_prompt_ids = base_prompt_ids[1:]
                if base_input_ids[0] == tokenizer.bos_token_id:
                    base_input_ids = base_input_ids[1:]

        base_input_ids = torch.tensor(base_input_ids, dtype=torch.long)
        base_prompt_ids = torch.tensor(base_prompt_ids, dtype=torch.long)
        base_prompt_length = len(base_prompt_ids)

        output_ids = copy.deepcopy(base_input_ids)
        output_ids[:base_prompt_length] = IGNORE_INDEX

        if positions == "all_prompt":
            intervention_locations = get_intervention_locations(
                positions="all", last_position=base_prompt_length
            )
        elif positions == "all":
            intervention_locations = get_intervention_locations(
                positions="all", last_position=len(base_input_ids)
            )
            if prefix_length is not None:
                intervention_locations = [
                    [loc for loc in locs if loc >= prefix_length]
                    for locs in intervention_locations
                ]
        else:
            intervention_locations = get_intervention_locations(
                positions=positions, last_position=base_prompt_length
            )

        all_base_input_ids.append(base_input_ids)
        all_output_ids.append(output_ids)
        all_intervention_locations.append(intervention_locations[0])

    train_dataset = Dataset.from_dict(
        {
            "input_ids": all_base_input_ids,
            "labels": all_output_ids,
            "intervention_locations": all_intervention_locations,
        }
    )
    train_dataset.set_format(type="torch")

    data_collator_fn = ReftDataCollator(
        DefaultDataCollator(), tokenizer=tokenizer, padding_side=padding_side
    )
    return dict(train_dataset=train_dataset, data_collator=data_collator_fn)


@dataclass
class ReftPreferenceDataCollator:
    """
    Args:
        padding_side: Use right padding for training and left padding for inference.
    """

    data_collator: DefaultDataCollator
    tokenizer: PreTrainedTokenizer
    padding_side: Literal["left", "right"] = "right"

    def __call__(self, instances: List[Dict[str, Any]]):
        max_seq_len = -1
        for inst in instances:
            for k in ("positive", "negative"):
                max_seq_len = max(max_seq_len, len(inst[f"{k}_input_ids"]))

        for inst in instances:
            for key in ("positive", "negative"):
                non_pad_len = len(inst[f"{key}_input_ids"])

                intervention_location_paddings = torch.tensor(
                    [
                        LOCATION_PAD
                        for _ in range(max_seq_len - len(inst[f"{key}_intervention_locations"]))
                    ]
                )
                if self.padding_side == "right":
                    inst[f"{key}_intervention_locations"] = torch.cat(
                        [inst[f"{key}_intervention_locations"], intervention_location_paddings]
                    ).long()
                else:
                    inst[f"{key}_intervention_locations"] = torch.cat(
                        [
                            intervention_location_paddings,
                            inst[f"{key}_intervention_locations"] + max_seq_len - non_pad_len,
                        ]
                    ).long()

                input_ids_paddings = torch.tensor(
                    [self.tokenizer.pad_token_id for _ in range(max_seq_len - non_pad_len)]
                )
                if self.padding_side == "right":
                    inst[f"{key}_input_ids"] = torch.cat(
                        [inst[f"{key}_input_ids"], input_ids_paddings]
                    ).int()
                else:
                    inst[f"{key}_input_ids"] = torch.cat(
                        [input_ids_paddings, inst[f"{key}_input_ids"]]
                    ).int()

                labels_paddings = torch.tensor(
                    [IGNORE_INDEX for _ in range(max_seq_len - non_pad_len)]
                )
                if self.padding_side == "right":
                    inst[f"{key}_labels"] = torch.cat([inst[f"{key}_labels"], labels_paddings]).long()
                else:
                    inst[f"{key}_labels"] = torch.cat([labels_paddings, inst[f"{key}_labels"]]).long()

                inst[f"{key}_attention_mask"] = (
                    inst[f"{key}_input_ids"] != self.tokenizer.pad_token_id
                ).int()

        batch = self.data_collator(instances)
        return batch


def curate_preference_training_data(
    tokenizer: PreTrainedTokenizer,
    positive_inputs: list,
    positive_outputs: list,
    negative_inputs: list,
    negative_outputs: list,
    positions: str = "f4",
    padding_side: Literal["right", "left"] = "right",
    prompt_max_length: int = 4096,
    max_length: int = 4096,
    eos_token: str = None,
    prefix_length: int = None,
):
    """
    Preference data curation for both training and inference,
    with intervention locations.

    `x_pos + y_pos` is preferred over `x_neg + y_neg`.

    Args:
        positive_inputs: x_pos.
        positive_outputs: y_pos.
        negative_inputs: x_neg.
        negative_outputs: y_neg.
        positions: f2, l3, f4+l5, all_prompt, all.
            * f2: First 2 prompt tokens.
            * l3: Last 3 prompt tokens.
            * f4+l5: First 4 and last 5 prompt tokens.
            * all_prompt: All prompt tokens.
            * all: All or prompt + response.
        padding_side: "right" for training and "left" for inference.
            Right padding is essential for intervention training,
            since we do not want interventions on
            padding tokens to affect actual tokens.
        eos_token: Not append EOS token if None.

    Returns:
        {"train_dataset": ..., "data_collator": ...}
    """
    assert (
        len(positive_inputs) == len(positive_outputs)
        and len(positive_inputs) == len(negative_inputs)
        and len(negative_inputs) == len(negative_outputs)
    ), "There should be the same number of positive/negative pairs."

    all_pos_input_ids = []
    all_pos_output_ids = []
    all_pos_intervention_locations = []
    all_neg_input_ids = []
    all_neg_output_ids = []
    all_neg_intervention_locations = []

    n = len(positive_inputs)
    disable_tqdm = False if n > 1000 else True # Silence if dataset is small
    for i in tqdm(range(n), desc="Preprocessing data", disable=disable_tqdm):
        def get_ids_and_locations(_input, _output):
            base_prompt = _input
            base_input = base_prompt + _output
            if eos_token is not None:
                base_input += eos_token

            base_prompt_ids = tokenizer(
                base_prompt,
                max_length=prompt_max_length,
                truncation=True,
            )["input_ids"]
            base_input_ids = tokenizer(
                base_input,
                max_length=max_length,
                truncation=True,
            )["input_ids"]

            if tokenizer.bos_token is not None:
                # Prepend bos token if chat template needs one; remove otherwise
                if (
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": "foo"}], tokenize=True
                    )[0]
                    == tokenizer.bos_token_id
                ):
                    if base_prompt_ids[0] != tokenizer.bos_token_id:
                        base_prompt_ids = [tokenizer.bos_token_id] + base_prompt_ids
                    if base_input_ids[0] != tokenizer.bos_token_id:
                        base_input_ids = [tokenizer.bos_token_id] + base_input_ids
                else:
                    if base_prompt_ids[0] == tokenizer.bos_token_id:
                        base_prompt_ids = base_prompt_ids[1:]
                    if base_input_ids[0] == tokenizer.bos_token_id:
                        base_input_ids = base_input_ids[1:]

            base_input_ids = torch.tensor(base_input_ids, dtype=torch.long)
            base_prompt_ids = torch.tensor(base_prompt_ids, dtype=torch.long)
            base_prompt_length = len(base_prompt_ids)

            output_ids = copy.deepcopy(base_input_ids)
            output_ids[:base_prompt_length] = IGNORE_INDEX

            if positions == "all_prompt":
                intervention_locations = get_intervention_locations(
                    positions="all", last_position=base_prompt_length
                )
            elif positions == "all":
                intervention_locations = get_intervention_locations(
                    positions="all", last_position=len(base_input_ids)
                )
                if prefix_length is not None:
                    intervention_locations = [
                        [loc for loc in locs if loc >= prefix_length]
                        for locs in intervention_locations
                    ]
            else:
                intervention_locations = get_intervention_locations(
                    positions=positions, last_position=base_prompt_length
                )
            return base_input_ids, output_ids, intervention_locations

        pos_input = positive_inputs[i]
        pos_output = positive_outputs[i]
        pos_ids, pos_labels, pos_locs = get_ids_and_locations(pos_input, pos_output)
        all_pos_input_ids.append(pos_ids)
        all_pos_output_ids.append(pos_labels)
        all_pos_intervention_locations.append(pos_locs[0])

        neg_input = negative_inputs[i]
        neg_output = negative_outputs[i]
        neg_ids, neg_labels, neg_locs = get_ids_and_locations(neg_input, neg_output)
        all_neg_input_ids.append(neg_ids)
        all_neg_output_ids.append(neg_labels)
        all_neg_intervention_locations.append(neg_locs[0])

    train_dataset = Dataset.from_dict(
        {
            "positive_input_ids": all_pos_input_ids,
            "positive_labels": all_pos_output_ids,
            "positive_intervention_locations": all_pos_intervention_locations,
            "negative_input_ids": all_neg_input_ids,
            "negative_labels": all_neg_output_ids,
            "negative_intervention_locations": all_neg_intervention_locations,
        }
    )
    train_dataset.set_format(type="torch")

    data_collator_fn = ReftPreferenceDataCollator(
        DefaultDataCollator(), tokenizer=tokenizer, padding_side=padding_side
    )
    return dict(train_dataset=train_dataset, data_collator=data_collator_fn)


@dataclass
class ReftVanillaDataCollator:
    """
    Args:
        padding_side: Use right padding for training and left padding for inference.
    """

    data_collator: DefaultDataCollator
    tokenizer: PreTrainedTokenizer
    padding_side: Literal["left", "right"] = "right"

    def __call__(self, instances: List[Dict[str, Any]]):
        max_seq_len = -1
        for inst in instances:
            max_seq_len = max(max_seq_len, len(inst["input_ids"]))

        for inst in instances:
            non_pad_len = len(inst["input_ids"])

            input_ids_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - non_pad_len)]
            )
            if self.padding_side == "right":
                inst["input_ids"] = torch.cat(
                    [inst["input_ids"], input_ids_paddings]
                ).int()
            else:
                inst["input_ids"] = torch.cat(
                    [input_ids_paddings, inst["input_ids"]]
                ).int()

            labels_paddings = torch.tensor(
                [IGNORE_INDEX for _ in range(max_seq_len - non_pad_len)]
            )
            if self.padding_side == "right":
                inst["labels"] = torch.cat([inst["labels"], labels_paddings]).long()
            else:
                inst["labels"] = torch.cat([labels_paddings, inst["labels"]]).long()

            inst["attention_mask"] = (
                inst["input_ids"] != self.tokenizer.pad_token_id
            ).int()

        batch = self.data_collator(instances)
        return batch


def curate_vanilla_training_data(
    tokenizer: PreTrainedTokenizer,
    inputs: list,
    outputs: list,
    padding_side: Literal["right", "left"] = "right",
    prompt_max_length: int = 4096, max_length: int = 4096,
    eos_token: str = None,
):
    """
    Data curation for both training and inference.

    Args:
        padding_side: "right" for training and "left" for inference.
            Right padding is essential for intervention training,
            since we do not want interventions on
            padding tokens to affect actual tokens.
        eos_token: Not append EOS token if None.

    Returns:
        {"train_dataset": ..., "data_collator": ...}
    """
    all_base_input_ids = []
    all_output_ids = []

    disable_tqdm = False if len(inputs) > 1000 else True # Silence if dataset is small
    for i in tqdm(range(len(inputs)), desc="Preprocessing data", disable=disable_tqdm):
        _input = inputs[i]
        _output = outputs[i]

        base_prompt = _input
        base_input = base_prompt + _output
        if eos_token is not None:
            base_input += eos_token

        base_prompt_ids = tokenizer(
            base_prompt,
            max_length=prompt_max_length,
            truncation=True,
        )["input_ids"]
        base_input_ids = tokenizer(
            base_input,
            max_length=max_length,
            truncation=True,
        )["input_ids"]

        if tokenizer.bos_token is not None:
            # Prepend bos token if chat template needs one; remove otherwise
            if (
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": "foo"}], tokenize=True
                )[0]
                == tokenizer.bos_token_id
            ):
                if base_prompt_ids[0] != tokenizer.bos_token_id:
                    base_prompt_ids = [tokenizer.bos_token_id] + base_prompt_ids
                if base_input_ids[0] != tokenizer.bos_token_id:
                    base_input_ids = [tokenizer.bos_token_id] + base_input_ids
            else:
                if base_prompt_ids[0] == tokenizer.bos_token_id:
                    base_prompt_ids = base_prompt_ids[1:]
                if base_input_ids[0] == tokenizer.bos_token_id:
                    base_input_ids = base_input_ids[1:]

        base_input_ids = torch.tensor(base_input_ids, dtype=torch.long)
        base_prompt_ids = torch.tensor(base_prompt_ids, dtype=torch.long)
        base_prompt_length = len(base_prompt_ids)

        output_ids = copy.deepcopy(base_input_ids)
        output_ids[:base_prompt_length] = IGNORE_INDEX

        all_base_input_ids.append(base_input_ids)
        all_output_ids.append(output_ids)

    train_dataset = Dataset.from_dict(
        {
            "input_ids": all_base_input_ids,
            "labels": all_output_ids,
        }
    )
    train_dataset.set_format(type="torch")

    data_collator_fn = ReftVanillaDataCollator(
        DefaultDataCollator(), tokenizer=tokenizer, padding_side=padding_side
    )
    return dict(train_dataset=train_dataset, data_collator=data_collator_fn)
