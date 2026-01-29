import copy
from datasets import Dataset
from loguru import logger
from tqdm import tqdm
from typing import Dict, List, Literal, Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    PreTrainedModel,
)

from .constants import IGNORE_INDEX, DEFAULT_PAD_TOKEN


def disable_model_gradients(model: nn.Module):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def load_hf_model_tokenizer(
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device="cpu",
    padding_side: Literal["left", "right"] = "right",
    disable_gradients=True,
    use_cache=True,
    load_in_4bit=False,
    load_model=True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Args:
        padding_side: Right padding when training; left padding for predicting
            latents or steering.
        use_cache: Our implementation allows seamless integration of KV cache
            at inference time.
            This option does not affect training.
        load_in_4bit: **[[DEPRECATED]]**.

    :return hf_model, tokenizer:
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    tokenizer.add_bos_token = True
    need_resize = False
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            logger.warning(f"Using unk_token `{tokenizer.unk_token}` as pad token.")
            tokenizer.pad_token = tokenizer.unk_token
        else:
            logger.warning("No pad_token or unk_token; adding one manually, requires resizing.")
            tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN})
            need_resize = True
    tokenizer.model_max_length = 16 * 1024
    tokenizer.padding_side = padding_side

    if not load_model:
        return None, tokenizer

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager", # we don't use flash attention
        use_cache=use_cache,
    )

    if need_resize:
        hf_model.resize_token_embeddings(len(tokenizer))

    if disable_gradients:
        disable_model_gradients(hf_model)

    return hf_model, tokenizer


@torch.no_grad()
def set_decoder_norm_to_unit_norm(lin: torch.nn.Module):
    assert hasattr(lin, "weight") and lin.weight is not None, (
        "Decoder weight was not initialized."
    )

    eps = torch.finfo(lin.weight.dtype).eps
    if lin.weight.data.shape[0] > lin.weight.data.shape[1]:
        dim = 0
    else:
        dim = 1
    norm = torch.norm(lin.weight.data, dim=dim, keepdim=True)
    lin.weight.data /= norm + eps

