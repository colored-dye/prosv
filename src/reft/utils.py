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


def spawn_edit_fn(
    lora: Dict[str, nn.Module],
    target_modules: List[str],
    prefix_length=0,
    mode: Literal["prefix", "suffix"] = "prefix",
    debug=False,
):
    """
    [[DEPRECATED]]

    Return a function for `Trace`/`TraceDict` to edit activations.

    Edit output representations of a module **in-place**.

    Args:
        prefix_length: Prefix is skipped and not edited.
        mode: "prefix" mode only edits prefix, while "suffix" mode 
            only edits suffix and prefix is skipped.
    """
    def _edit_fn(output, layer, inputs):
        if layer not in target_modules:
            return output

        rest = None
        if isinstance(output, tuple):
            output, *rest = output

        if mode == "suffix": # Keep prefix intact
            if debug:
                logger.warning("Suffix mode intervention")

            prefix = output[:, :prefix_length].clone()
            y = lora[layer](output)
            y[:, :prefix_length] = prefix
        elif mode == "prefix": # Keep suffix intact
            assert len(output.shape) == 3
            if output.size(1) > 1: # prompt, not response
                if debug:
                    logger.warning("Prefix mode intervention")

                y = output.clone()
                if isinstance(prefix_length, Iterable):
                    if debug:
                        logger.warning("prefix length is iterable")

                    for i in range(y.size(0)):
                        prefix_len = prefix_length[i]
                        prefix = output[i, :prefix_len]
                        prefix_intervened = lora[layer](prefix)
                        y[i, :prefix_len] = prefix_intervened
                else:
                    if debug:
                        logger.warning("Prefix length is an integer")

                    prefix = output[:, :prefix_length]
                    prefix_intervened = lora[layer](prefix)
                    y[:, :prefix_length] = prefix_intervened
            else:
                if debug:
                    logger.warning("No intervention")

                y = output
        else:
            raise ValueError(f"Unknown mode: `{mode}`")

        if rest is not None:
            return y, *rest
        else:
            return y

    return _edit_fn


def spawn_edit_fn_separate(
    lora: Dict[str, nn.Module],
    input_target_modules: List[str],
    edit_target_modules: List[str],
    prefix_length=0,
):
    """
    [[DEPRECATED]]

    Return a function for `Trace`/`TraceDict` to edit activations.

    Get inputs from the inputs of a module and 
    use it to edit the outputs of another module.

    Let model be f(x) with residual connection: y = x + f(x).
    Then spawn_edit_fn_separate(f) substitutes f(x): g(f(x)+x) - x,
    where g(.) is an intervention.

    Args:
        prefix_length: Prefix is skipped and not edited.
    """
    cache_inputs = None

    def _edit_fn(output, layer, inputs):
        nonlocal cache_inputs

        if layer not in input_target_modules and layer not in edit_target_modules:
            return output

        if layer in input_target_modules:
            if isinstance(inputs, tuple):
                cache_inputs = inputs[0]
            else:
                cache_inputs = inputs
            return output

        elif layer in edit_target_modules:
            rest = None
            if isinstance(output, tuple):
                output, *rest = output

            prefix = output[:, :prefix_length].clone()
            y = lora[layer](cache_inputs + output)
            y[:, :prefix_length] = prefix

            # This is essential, since we edit the residual:
            # y = (W_reft + I) (x + output).
            # But un-intervened outputs is: output.
            # Therefore the extra x should be subtracted from y.
            y = y - cache_inputs

            if rest is not None:
                return y, *rest
            else:
                return y

    return _edit_fn


def curate_data(
    tokenizer: PreTrainedTokenizer,
    inputs: list,
    outputs: list,
    padding_side: Literal["right", "left"] = "right",
    prompt_max_length: int = 4096,
    max_length: int = 4096,
    eos_token: str = None,
    model=None,
):
    """
    **[[DEPRECATED]]**

    Data curation for both training and inference.

    Args:
        padding_side: "right" for training and "left" for inference.
            Right padding is essential for intervention training,
            since we do not want interventions on
            padding tokens to affect actual tokens.
        eos_token: Not append EOS token if None.
    """
    logger.error("Do not use this function!")
    raise ValueError("Do not use this function!")

    all_base_input_ids, all_output_ids = [], []
    all_prompt_lengths = []
    for i in tqdm(range(len(inputs)), desc="Preprocessing data"):
        _input = inputs[i]
        _output = outputs[i]

        base_prompt = _input
        base_input = base_prompt + _output
        if eos_token is not None:
            base_input += eos_token

        # tokenize
        base_prompt_ids = tokenizer(
            base_prompt,
            max_length=prompt_max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        base_input_ids = tokenizer(
            base_input,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]
        output_ids = copy.deepcopy(base_input_ids)
        output_ids[:base_prompt_length] = IGNORE_INDEX

        all_base_input_ids.append(base_input_ids)
        all_output_ids.append(output_ids)
        all_prompt_lengths.append(len(base_prompt_ids))

    train_dataset = Dataset.from_dict(
        {
            "input_ids": all_base_input_ids,
            "labels": all_output_ids,
            "prompt_lengths": all_prompt_lengths,
        }
    )

    tokenizer.padding_side = padding_side
    data_collator_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        padding="longest",
    )
    return dict(train_dataset=train_dataset, data_collator=data_collator_fn)


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
    yarn_factor=None,
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

    # if 'meta-llama/llama-3.1-8b-instruct'.lower() in model_name_or_path.lower():
    #     logger.warning("Replacing chat template")
    #     chat_template = "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"
    #     tokenizer.chat_template = chat_template

    # We no longer use custom quant configs for two reasons:
    # 1. It conflicts with built-in quant configs;
    # 2. It hurts model performance, resulting in non-negligible changes in steering scores.
    # bnb_config = None
    # if load_in_4bit:
    #     logger.warning("Loading model with 4 bit quantization.")
    #     bnb_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=dtype,
    #     )

    config = AutoConfig.from_pretrained(model_name_or_path)
    if config.model_type == "gemma3":
        try:
            from transformers import Gemma3ForCausalLM
        except Exception as e:
            raise ImportError(e)
        hf_model = Gemma3ForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager", # we don't use flash attention
            use_cache=use_cache,
        )
    else:
        if yarn_factor is not None:
            logger.warning("YaRN context scaling.")
            config.rope_scaling = {
                "type": "yarn",
                "factor": yarn_factor,
                "original_max_position_embeddings": config.max_position_embeddings,
            }

            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                config=config,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
                attn_implementation="eager", # we don't use flash attention
                # use_cache=use_cache,
                # quantization_config=bnb_config, 
            )
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
                attn_implementation="eager", # we don't use flash attention
                use_cache=use_cache,
                # quantization_config=bnb_config, 
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


def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Taken from https://github.com/stanfordnlp/axbench/blob/main/axbench/models/preference_model.py#L38

    Ref of Eric's repo: 
    https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L90

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != IGNORE_INDEX)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == IGNORE_INDEX] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    gemma: float,
                    simpo_scaler: float,
                    winning_lens: torch.LongTensor,
                    losing_lens: torch.LongTensor,
                    label_smoothing: float = 0.0,
                    loss_type: str = "dpo",
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Taken from https://github.com/stanfordnlp/axbench/blob/main/axbench/models/preference_model.py#L69

    Ref of Eric's repo: 
    https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L45

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        loss_type: different preference loss functions.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    ref_logratios_reverse = reference_rejected_logps - reference_chosen_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if loss_type == "ipo":
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    elif loss_type == "dpo":
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    elif loss_type == "simpo":
        losses = -F.logsigmoid((beta / winning_lens) * policy_chosen_logps - (beta / losing_lens) * policy_rejected_logps - gemma)
    elif loss_type == "scaled_simpo":
        scaled_policy_chosen_logps = (
            torch.max(
                ref_logratios_reverse * simpo_scaler,
                torch.ones_like(ref_logratios_reverse),
            )
            / winning_lens
        ) * policy_chosen_logps
        scaled_policy_rejected_logps = (1.0 / losing_lens) * policy_rejected_logps
        losses = -F.logsigmoid(scaled_policy_chosen_logps - scaled_policy_rejected_logps)
        """
        negative steering:

        input: steering prefix + original instruction
        winning output: original response
        losing output: steered response

        scaler = p_ref(losing output) - p_ref(winning output)
        losses = -F.logsigmoid(
            (torch.max(scaler, 1) / winning_lens) * policy_chosen_logps - (1.0 / losing_lens) * policy_rejected_logps)
        """
    elif loss_type == "apo_zero":
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        losses = -F.logsigmoid(beta * chosen_logratios) + F.logsigmoid(beta * rejected_logratios) 
    else:
        raise ValueError(f"Loss type {loss_type} not supported")

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards
