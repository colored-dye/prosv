from collections import defaultdict
from datasets import load_dataset, Dataset
import random
import re
from tqdm import tqdm
from typing import Iterable, Dict, List

from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
)

from reft.dataset import curate_training_data, curate_vanilla_training_data
from reft.intervenable import IntervenableModel, InterventionMode


def format_options(choices: Iterable[str]):
    """Format multiple choice options."""
    formatted = "\nOptions:\n"
    for i, choice in enumerate(choices):
        formatted += f"{chr(65+i)}. {choice}\n"
    return formatted


def shuffle_choices_and_labels(choices: Iterable[str], labels, seed):
    """Shuffle choices and their corresponding labels together."""
    combined = list(zip(choices, labels))
    random.Random(seed).shuffle(combined)
    shuffled_choices, shuffled_labels = zip(*combined)
    return list(shuffled_choices), list(shuffled_labels)


def format_prompt(
    question: str,
    choices: Iterable[str],
    few_shot_prompt: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
):
    """Format the question into a templated prompt with few-shot examples."""
    prompt = f"Question: {question}\n"
    prompt += format_options(choices)
    prompt += "\nAnswer:"
    ids = tokenizer.apply_chat_template(
        few_shot_prompt + [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if tokenizer.bos_token is not None and ids[0] == tokenizer.bos_token_id:
        ids = ids[1:]
    return tokenizer.decode(ids)


def mmlu_eval(
    intervenable: IntervenableModel,
    positions: str,
    tokenizer: PreTrainedTokenizer,
    hf_dataset_path: str,
    logger,
    seed: int,
    num_shots: int,
    batch_size: int,
    max_new_tokens: int,
    no_interventions=False,
    steering_prompt: str = None,
):
    """MMLU (cais/mmlu)."""
    MMLU_SUBJECTS = [ "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology", "college_chemistry", "college_computer_science", "college_mathematics", "college_medicine", "college_physics", "computer_security", "conceptual_physics", "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science", "high_school_european_history", "high_school_geography", "high_school_government_and_politics", "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics", "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality", "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law", "professional_medicine", "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions", ]

    def _get_few_shot_prompt(
        dataset: Dataset,
        num_shots: int,
        seed: int,
    ):
        random.seed(seed)
        selected_examples = random.sample(list(dataset), num_shots)
        examples = []

        for item in selected_examples:
            question = item["question"]
            choices = item["choices"]
            correct_idx = item["answer"]
            answer = chr(65 + correct_idx)

            prompt = f"Question: {question}\n"
            prompt += format_options(choices)
            prompt += "\nAnswer:"

            # "The answer is ..." prefix is essential.
            msgs = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"The answer is {answer}."},
            ]
            examples.extend(msgs)

        return examples

    total_acc = 0
    results = defaultdict(list)
    for subject_index, subject in enumerate(tqdm(MMLU_SUBJECTS, desc="MMLU")):
        dev_set = load_dataset(hf_dataset_path, subject, split="dev")
        test_set = load_dataset(hf_dataset_path, subject, split="test")

        prompts = []
        answers = []
        for idx, item in enumerate(test_set):
            few_shot_prompt = _get_few_shot_prompt(
                dataset=dev_set,
                num_shots=num_shots,
                seed=seed+subject_index,
            )
            prompt = format_prompt(
                question=item['question'],
                choices=item['choices'],
                few_shot_prompt=few_shot_prompt,
                tokenizer=tokenizer,
            )
            answer = chr(65 + item['answer'])

            prompts.append(prompt)
            answers.append(answer)
            results["original_prompt"].append(item['question'])
            results["answer"].append(answer)
            results["metadata"].append(f"MMLU_{subject}")

        data_module = curate_training_data(
            tokenizer=tokenizer,
            inputs=prompts,
            outputs=["" for _ in prompts],
            positions=positions,
            padding_side="left",
        )
        train_set, collator = data_module["train_dataset"], data_module["data_collator"]
        dataloader = DataLoader(
            train_set, collate_fn=collator, batch_size=batch_size, shuffle=False
        )
        # logger.warning("**" + tokenizer.decode(train_set[0]["input_ids"]) + "**")

        all_responses = []
        desc = f"Steering on subject [{subject_index}/{len(MMLU_SUBJECTS)}]"
        if no_interventions:
            desc += " (Vanilla)"
        for batch in tqdm(dataloader, desc=desc):
            locations = batch["intervention_locations"]
            if positions == "all": # Special treatment for full-sequence generation
                intervention_mode = InterventionMode.ALL_GENERATION
            else:
                intervention_mode = InterventionMode.PROMPT_ONLY

            if hasattr(intervenable, "model") and intervenable.model.generation_config is not None:
                _eos_ids = intervenable.model.generation_config.eos_token_id
                stop_strings = (
                    tokenizer.batch_decode(_eos_ids)
                    if isinstance(_eos_ids, list)
                    else [tokenizer.decode(_eos_ids)]
                )
                # stop_strings.append('\n')
            elif hasattr(intervenable, "config"):
                _eos_ids = intervenable.generation_config.eos_token_id
                stop_strings = (
                    tokenizer.batch_decode(_eos_ids)
                    if isinstance(_eos_ids, list)
                    else [tokenizer.decode(_eos_ids)]
                )
            else:
                stop_strings = [tokenizer.eos_token]
            if no_interventions:
                outputs = intervenable.model.generate(
                    input_ids=batch["input_ids"].to(intervenable.device),
                    attention_mask=batch["attention_mask"].to(intervenable.device),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    stop_strings=stop_strings,
                    tokenizer=tokenizer,
                )
            else:
                outputs = intervenable.generate(
                    locations=locations,
                    intervention_mode=intervention_mode,
                    input_ids=batch["input_ids"].to(intervenable.device),
                    attention_mask=batch["attention_mask"].to(intervenable.device),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    stop_strings=stop_strings,
                    tokenizer=tokenizer,
                )

            out_text = tokenizer.batch_decode(
                outputs[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
            )
            logger.warning("**" + out_text[0] + "**")
            all_responses.extend(out_text)

        results["steered_generation"].extend(all_responses)

        acc = 0
        for r, a in zip(all_responses, answers):
            if a in r:
                acc += 1
        total_acc += acc
        logger.warning(f"Subject [{subject_index}/{len(MMLU_SUBJECTS)}] `{subject}` acc: {acc/len(answers)*100:.3f}%")

    total_acc /= len(results['metadata'])
    logger.warning("=================================")
    logger.warning(f"Total acc: {total_acc*100:.3f}%")
    logger.warning("=================================")

    return results


def tinymmlu_eval(
    intervenable: IntervenableModel,
    positions: str,
    tokenizer: PreTrainedTokenizer,
    hf_dataset_path: str,
    logger,
    seed: int,
    num_shots: int,
    batch_size: int,
    max_new_tokens: int,
    no_interventions=False,
    steering_prompt: str = None,
):
    """tinyMMLU (tinyBenchmarks/tinyMMLU)."""
    results = defaultdict(list)
    test_set = load_dataset(hf_dataset_path, split="test")

    prompts = []
    answers = []
    for rec in test_set:
        pairs = rec['input_formatted'].split('Answer:')
        task_desc = re.findall(r'The following .*\n', pairs[0])[0].strip()
        msgs = []
        q = pairs[0].replace(task_desc, '').strip()
        for i, p in enumerate(pairs[1:-1]):
            a = p[1]
            msgs.append({'role': 'user', 'content': task_desc + "\n\n" + q + "\nAnswer:"})
            msgs.append({'role': 'assistant', 'content': f"The answer is: {a}."})
            q = p[4:].strip()
        final_q = task_desc + "\n\n" + q + "\nAnswer:"
        if steering_prompt is not None:
            final_q = steering_prompt + "\n\n" + final_q
        msgs.append({'role': 'user', 'content': final_q})
        results["original_prompt"].append(q)

        ids = tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
        )
        if tokenizer.bos_token is not None and ids[0] == tokenizer.bos_token_id:
            ids = ids[1:]
        prompt = tokenizer.decode(ids)
        prompts.append(prompt)

        ans = chr(65 + rec['answer'])
        answers.append(ans)

    if no_interventions or steering_prompt is not None:
        data_module = curate_vanilla_training_data(
            tokenizer=tokenizer,
            inputs=prompts,
            outputs=["" for _ in prompts],
            padding_side="left",
        )
    else:
        data_module = curate_training_data(
            tokenizer=tokenizer,
            inputs=prompts,
            outputs=["" for _ in prompts],
            positions=positions,
            padding_side="left",
        )
    train_set, collator = data_module["train_dataset"], data_module["data_collator"]
    dataloader = DataLoader(
        train_set, collate_fn=collator, batch_size=batch_size, shuffle=False
    )
    logger.warning("**" + tokenizer.decode(train_set[0]["input_ids"]) + "**")

    desc = "Steering on tinyMMLU"
    if no_interventions:
        desc += " (Vanilla)"
    if steering_prompt is not None:
        desc += " (Prompt steering)"

    all_responses = []
    for batch in tqdm(dataloader, desc=desc):
        if no_interventions or steering_prompt is not None:
            assert isinstance(intervenable, PreTrainedModel)
            outputs = intervenable.generate(
                input_ids=batch["input_ids"].to(intervenable.device),
                attention_mask=batch["attention_mask"].to(intervenable.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        else:
            locations = batch["intervention_locations"]
            if positions == "all": # Special treatment for full-sequence generation
                intervention_mode = InterventionMode.ALL_GENERATION
            else:
                intervention_mode = InterventionMode.PROMPT_ONLY

            outputs = intervenable.generate(
                locations=locations,
                intervention_mode=intervention_mode,
                input_ids=batch["input_ids"].to(intervenable.device),
                attention_mask=batch["attention_mask"].to(intervenable.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        out_text = tokenizer.batch_decode(
            outputs[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
        )
        logger.warning("**" + out_text[0] + "**")
        all_responses.extend(out_text)
        IntervenableModel.clear_cache()

    results['answer'] = answers
    results['metadata'] = ['tinyMMLU'] * len(prompts)
    results["steered_generation"] = all_responses

    acc = 0
    for pred, ans in zip(results['steered_generation'], results['answer']):
        if ans in pred:
            acc += 1
    acc /= len(prompts)

    logger.warning("=================================")
    logger.warning(f"Total acc: {acc*100:.3f}%")
    logger.warning("=================================")

    return results

def tinygsm8k_eval(
    intervenable: IntervenableModel,
    positions: str,
    tokenizer: PreTrainedTokenizer,
    hf_dataset_path: str,
    logger,
    seed: int,
    num_shots: int,
    batch_size: int,
    max_new_tokens: int,
    no_interventions=False,
    steering_prompt: str = None,
):
    """tinyGSM8K (tinyBenchmarks/tinyGSM8k)."""
    results = defaultdict(list)
    test_set = load_dataset(hf_dataset_path, split="test")

    prompts = []
    answers = []
    for rec in test_set:
        pairs = rec['input_formatted'].split('\n\n')
        msgs = []
        for p in pairs[:-1]:
            few_shot_q = re.findall(r'Question:.*', p)
            few_shot_a = re.findall(r'Answer: .*', p, re.DOTALL)
            msgs.append({'role': 'user', 'content': few_shot_q[0]})
            msgs.append({'role': 'assistant', 'content': few_shot_a[0]})
        q = re.findall(r'Question:.*', pairs[-1])[0]
        if steering_prompt is not None:
            q = steering_prompt + "\n\n" + q
        msgs.append({'role': 'user', 'content': q})
        results["original_prompt"].append(q.replace("Question: ", ""))

        ids = tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
        )
        if tokenizer.bos_token is not None and ids[0] == tokenizer.bos_token_id:
            ids = ids[1:]
        prompt = tokenizer.decode(ids)
        prompts.append(prompt)

        matched = re.findall(r'#### (.*)', rec['answer'], re.DOTALL)
        ans = matched[0].strip()
        answers.append(ans)

    if no_interventions or steering_prompt is not None:
        data_module = curate_vanilla_training_data(
            tokenizer=tokenizer,
            inputs=prompts,
            outputs=["" for _ in prompts],
            padding_side="left",
        )
    else:
        data_module = curate_training_data(
            tokenizer=tokenizer,
            inputs=prompts,
            outputs=["" for _ in prompts],
            positions=positions,
            padding_side="left",
        )
    train_set, collator = data_module["train_dataset"], data_module["data_collator"]
    dataloader = DataLoader(
        train_set, collate_fn=collator, batch_size=batch_size, shuffle=False
    )
    logger.warning("**" + tokenizer.decode(train_set[0]["input_ids"]) + "**")

    all_responses = []
    desc = "Steering on tinyGSM8K"
    if no_interventions:
        desc += " (Vanilla)"
    if steering_prompt is not None:
        desc += " (Prompt steering)"

    for batch in tqdm(dataloader, desc=desc):
        if no_interventions or steering_prompt is not None:
            outputs = intervenable.generate(
                input_ids=batch["input_ids"].to(intervenable.device),
                attention_mask=batch["attention_mask"].to(intervenable.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        else:
            locations = batch["intervention_locations"]
            if positions == "all": # Special treatment for full-sequence generation
                intervention_mode = InterventionMode.ALL_GENERATION
            else:
                intervention_mode = InterventionMode.PROMPT_ONLY

            outputs = intervenable.generate(
                locations=locations,
                intervention_mode=intervention_mode,
                input_ids=batch["input_ids"].to(intervenable.device),
                attention_mask=batch["attention_mask"].to(intervenable.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        out_text = tokenizer.batch_decode(
            outputs[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
        )
        logger.warning("**" + out_text[0] + "**")
        all_responses.extend(out_text)
        IntervenableModel.clear_cache()

    results['answer'] = answers
    results['metadata'] = ['tinyGSM8K'] * len(prompts)
    results["steered_generation"] = all_responses

    acc = 0
    for pred, ans in zip(results['steered_generation'], results['answer']):
        for pred in reversed(extract_answer_numbers(pred)):
            pred = round(pred, 3)
            ans = round(float(ans), 3)
            if abs(pred - ans) < 1e-3:
                acc += 1
                break
    acc /= len(prompts)

    logger.warning("=================================")
    logger.warning(f"Total acc: {acc*100:.3f}%")
    logger.warning("=================================")

    return results


def mbpp_eval(
    intervenable: IntervenableModel,
    positions: str,
    tokenizer: PreTrainedTokenizer,
    hf_dataset_path: str,
    logger,
    seed: int,
    num_shots: int,
    batch_size: int,
    max_new_tokens: int,
    no_interventions=False,
    steering_prompt: str = None,
):
    """MBPP (google-research-datasets/mbpp)."""
    MBPP_TEMPLATE = """You are an expert Python programmer, and here is your task:
{text}
Your code should pass these tests:
{test_list_0}
{test_list_1}
{test_list_2}"""

    results = defaultdict(list)

    dev_set = load_dataset(hf_dataset_path, split="prompt")
    test_set = load_dataset(hf_dataset_path, split="test")

    prompts = []
    for index, rec in enumerate(test_set):
        random.seed(seed+index)
        few_shot_examples = random.sample(dev_set.to_list(), num_shots)
        msgs = []
        for fse in few_shot_examples:
            prompt = MBPP_TEMPLATE.format(
                text=fse["text"],
                test_list_0=fse["test_list"][0],
                test_list_1=fse["test_list"][1],
                test_list_2=fse["test_list"][2],
            )
            response = "```python\n" + fse["code"] + "\n```"
            msgs.append({'role': 'user', 'content': prompt})
            msgs.append({'role': 'assistant', 'content': response})
        prompt = MBPP_TEMPLATE.format(
            text=rec["text"],
            test_list_0=rec["test_list"][0],
            test_list_1=rec["test_list"][1],
            test_list_2=rec["test_list"][2],
        )
        msgs.append({'role': 'user', 'content': prompt})

        ids = tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
        )
        if tokenizer.bos_token is not None and ids[0] == tokenizer.bos_token_id:
            ids = ids[1:]
        prompt = tokenizer.decode(ids)
        prompts.append(prompt)

    data_module = curate_training_data(
        tokenizer=tokenizer,
        inputs=prompts,
        outputs=["" for _ in prompts],
        positions=positions,
        padding_side="left",
    )
    train_set, collator = data_module["train_dataset"], data_module["data_collator"]
    dataloader = DataLoader(
        train_set, collate_fn=collator, batch_size=batch_size, shuffle=False
    )
    logger.warning("**" + tokenizer.decode(train_set[0]["input_ids"]) + "**")

    all_responses = []
    desc = "Steering on MBPP"
    if no_interventions:
        desc += " (Vanilla)"
    for batch in tqdm(dataloader, desc=desc):
        locations = batch["intervention_locations"]
        if positions == "all": # Special treatment for full-sequence generation
            intervention_mode = InterventionMode.ALL_GENERATION
        else:
            intervention_mode = InterventionMode.PROMPT_ONLY

        if no_interventions:
            outputs = intervenable.generate(
                input_ids=batch["input_ids"].to(intervenable.device),
                attention_mask=batch["attention_mask"].to(intervenable.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        else:
            outputs = intervenable.generate(
                locations=locations,
                intervention_mode=intervention_mode,
                input_ids=batch["input_ids"].to(intervenable.device),
                attention_mask=batch["attention_mask"].to(intervenable.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        out_text = tokenizer.batch_decode(
            outputs[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
        )
        logger.warning("**" + out_text[0] + "**")
        all_responses.extend(out_text)

    return results


def math_eval(
    intervenable: IntervenableModel,
    positions: str,
    tokenizer: PreTrainedTokenizer,
    hf_dataset_path: str,
    logger,
    seed: int,
    num_shots: int,
    batch_size: int,
    max_new_tokens: int,
    no_interventions=False,
    steering_prompt: str = None,
):
    """MATH500 (HuggingFaceH4/MATH-500)."""
    results = defaultdict(list)
    test_set = load_dataset(hf_dataset_path, split="test")

    return results


def humaneval_eval(
    intervenable: IntervenableModel,
    positions: str,
    tokenizer: PreTrainedTokenizer,
    hf_dataset_path: str,
    logger,
    seed: int,
    num_shots: int,
    batch_size: int,
    max_new_tokens: int,
    no_interventions=False,
    steering_prompt: str = None,
):
    """HumanEval (openai/openai_humaneval)."""
    results = defaultdict(list)
    test_set = load_dataset(hf_dataset_path, split="test")

    return results


def extract_answer_numbers(sentence: str) -> float:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py
    """
    sentence = sentence.replace(',', '')
    preds = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    def _parse(pred):
        if not pred:
            return float('inf')
        pred_answer = float(pred)
        if isinstance(pred_answer, str):
            try:
                pred_answer = float(pred_answer)
            except ValueError:
                pred_answer = float('inf')
        return pred_answer

    numbers = [_parse(n) for n in preds] if len(preds) > 0 else [float("inf")]
    return numbers


__all__ = ["mmlu_eval", "tinygsm8k_eval"]
