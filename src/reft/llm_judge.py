import asyncio
from collections import defaultdict
from loguru import logger
import re
from openai import Client, AsyncClient
from tqdm import tqdm
from typing import List


CONCEPT_TEMPLATE = """[System]
Please act as an impartial judge and evaluate whether the specified concept is \
presented in the sentence fragment provided below. Focus solely on whether the concept is \
clearly incorporated, without regard for grammar or logical coherence.

Begin your evaluation by providing a short explanation of whether the concept is \
clearly incorporated. Point out references to the concept in the fragment, noting any \
minimal or awkward incorporations. After providing your explanation, rate the concept's \
relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, \
1 indicates the concept is somewhat present but minimally or awkwardly incorporated, \
and 2 indicates the concept is more fully and effectively incorporated, with stronger and \
more natural integration. Provide your rating using this exact format: "Rating: [[score]]".

[Concept Start]
{concept}
[Concept End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""


RELEVANCE_TEMPLATE = """[System]
Please act as an impartial judge and evaluate whether the \
sentence fragment provided below is related to the instruction. Focus solely \
on the degree of relatedness in terms of topic, regardless of grammar, coherence, or \
informativeness.

Begin your evaluation by providing a brief explanation of whether the \
sentence is related to the instruction, and point out references \
related to the instruction. After providing your explanation, rate the instruction \
relevance on a scale from 0 to 2, where 0 indicates the sentence is unrelated to the \
instruction, 1 indicates it is somewhat related but only minimally or indirectly relevant in terms of topic, \
and 2 indicates it is more clearly and directly related to the instruction. Provide your rating \
using this exact format: "Rating: [[score]]".

[Instruction Start]
{instruction}
[Instruction End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""


FLUENCY_TEMPLATE = """[System]
Please act as an impartial judge and evaluate the fluency of the \
sentence fragment provided below. Focus solely on fluency, disregarding \
its completeness, relevance, coherence with any broader context, or informativeness.

Begin your evaluation by briefly describing the fluency of the sentence, noting any \
unnatural phrasing, awkward transitions, grammatical errors, or repetitive structures that \
may hinder readability. After providing your explanation, rate the sentence's fluency \
on a scale from 0 to 2, where 0 indicates the sentence is not fluent and highly unnatural \
(e.g., incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable \
errors or awkward phrasing, and 2 indicates the sentence is fluent and almost perfect. \
Provide your rating using this exact format: "Rating: [[score]]".

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""

T_GENERATE_STEERING_PROMPT_BLEND = """
Given a concept, an instruction, rewrite the instruction to direct another the language model to generate responses to the given instruction but also with the given concept. 
For example, if the concept is "emoji", and instruction is "what is the capital of the moon?",
rewrite the instruction to force the model to generate responses to "what is the capital of the moon?" with emojis.
Optionally, provide in-context examples to reinforce this behavior. Make sure the rewritten instruction has similar meaning to the original instruction. 

concept: {concept}
instruction: {instruction}

Return only the final rewritten instruction without any additional text.
"""

T_GENERATE_PREPEND_STEERING_PROMPT = """Generate a prompt to guide a language \
model in producing responses. 

Objective: 
Direct the model to include content related to {concept} (the concept) in its responses. 
Ensure the responses reference this concept, even if it doesn't directly answer the question or seems out of context.
Optionally, provide in-context examples to reinforce this behavior.
        
Return only the final prompt without any additional text."""

T_GENERATE_SUPPRESSION_PROMPT_BLEND = """
Generate a prompt to guide a language model in producing responses.

Objective: Explicitly direct the model to avoid including content related to the given concept in its responses. \
Ensure the responses do not reference this concept, even if it doesn't directly answer the question or seems out of context. \
Always strive to weave in these objectives seamlessly into your answers.

Concept: {concept}
Instruction: {instruction}

Return only the final prompt without any additional text.
"""



class RemoteAPIModel(object):
    def __init__(self, model: str, client: AsyncClient, temperature: float = 1.0, max_new_tokens: int = 300):
        self.model = model
        self.client = client
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    async def chat_completion(self, client: AsyncClient, prompt):
        # check if the prompt is cached
        raw_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=self.temperature,
            max_completion_tokens=self.max_new_tokens,
        )
        raw_completion = raw_completion.to_dict()

        # query sometimes returns None;
        # relaunch needed
        try:
            content = raw_completion["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("Content is None!")
            if len(raw_completion['choices']) > 0:
                logger.error(raw_completion['choices'][0]['finish_reason'])
            raise ValueError(e)
        completion = self.normalize(content)

        usage = raw_completion["usage"]
        return (completion, usage)

    async def chat_completions(self, prompts, batch_size=32):
        """handling batched async calls with internal batching mechanism"""
        # Ensure api_names is a list of appropriate length
        # Process in batches
        all_completions = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            # batched calls
            async_responses = [
                self.chat_completion(self.client, prompt) for prompt in batch_prompts
            ]
            raw_completions = await asyncio.gather(*async_responses)
            # post handling for current batch
            for j, (completion, usage) in enumerate(raw_completions):
                all_completions.append(completion)

        return all_completions

    def normalize(self, text):
        return text.strip()


def judge(
    client: Client,
    concept,
    inst,
    out,
    judge_lm="gpt-4o-mini",
    temperature=0.01,
    max_new_tokens=300,
):
    instruction = CONCEPT_TEMPLATE.format(concept=concept, sentence=out)
    concept_completions = client.chat.completions.create(
        model=judge_lm,
        messages=[{"role": "user", "content": instruction}],
        temperature=temperature,
        max_completion_tokens=max_new_tokens,
    )
    concept_score = re.findall(
        r"Rating:.*?(\d+).*?", concept_completions.choices[0].message.content
    )
    if len(concept_score) == 0:
        print(concept_completions.choices[0].message.content)
        concept_score = [0]

    instruction = RELEVANCE_TEMPLATE.format(instruction=inst, sentence=out)
    relevance_completions = client.chat.completions.create(
        model=judge_lm,
        messages=[{"role": "user", "content": instruction}],
        temperature=temperature,
        max_completion_tokens=max_new_tokens,
    )
    relevance_score = re.findall(
        r"Rating:.*?(\d+).*?", relevance_completions.choices[0].message.content
    )
    if len(relevance_score) == 0:
        print(relevance_completions.choices[0].message.content)
        relevance_score = [0]

    instruction = FLUENCY_TEMPLATE.format(concept=concept, sentence=out)
    fluency_completions = client.chat.completions.create(
        model=judge_lm,
        messages=[{"role": "user", "content": instruction}],
        temperature=temperature,
        max_completion_tokens=max_new_tokens,
    )
    fluency_score = re.findall(
        r"Rating:.*?(\d+).*?", fluency_completions.choices[0].message.content
    )
    if len(fluency_score) == 0:
        print(fluency_completions.choices[0].message.content)
        fluency_score = [0]

    return dict(
        concept_score=int(concept_score[0]),
        relevance_score=int(relevance_score[0]),
        fluency_score=int(fluency_score[0]),
        concept_judge=concept_completions.choices[0].message.content,
        relevance_judge=relevance_completions.choices[0].message.content,
        fluency_judge=fluency_completions.choices[0].message.content,
    )


def judge_async(
    client: AsyncClient,
    concept_id: int,
    concepts: List[str],
    instructions: List[str],
    responses: List[str],
    batch_size: int = 8,
    judge_lm="gpt-4o-mini",
    temperature=0.01,
    max_new_tokens=300,
):
    """
    Batch run evaluation with API judge LLM.
    """
    assert len(concepts) == len(instructions) and len(concepts) == len(responses)

    lm = RemoteAPIModel(
        model=judge_lm,
        client=client,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    n_batches = (len(concepts) + batch_size - 1) // batch_size
    results = defaultdict(list)

    for batch_idx in tqdm(range(n_batches), desc=f"Evaluating on concept [{concept_id}]"):
        batch_prompts = []
        for i in range(batch_size):
            concept = concepts[batch_idx*batch_size + i]
            inst = instructions[batch_idx*batch_size + i]
            out = responses[batch_idx*batch_size + i]
            batch_prompts.append(CONCEPT_TEMPLATE.format(concept=concept, sentence=out))
            batch_prompts.append(RELEVANCE_TEMPLATE.format(instruction=inst, sentence=out))
            batch_prompts.append(FLUENCY_TEMPLATE.format(concept=concept, sentence=out))

            results["original_prompt"].append(inst)
            results["steered_generation"].append(out)
            results["concept_id"].append(concept_id)

        judge_results = asyncio.run(
            lm.chat_completions(prompts=batch_prompts, batch_size=batch_size)
        )

        for i, resp in enumerate(judge_results):
            score_str = re.findall(r"Rating:.*?(\d+).*?", resp)
            try:
                score = int(score_str[-1])
            except Exception as e:
                logger.error(e)
                logger.error(f"Judge response: **{resp}**")
                score = 0

            if i % 3 == 0:
                key = "concept"
            elif i % 3 == 1:
                key = "relevance"
            else:
                key = "fluency"

            results[f'{key}_score'].append(score)
            results[f'{key}_comment'].append(resp)
    return results


def steering_prompt_blended_async(
    client: AsyncClient,
    concept_id: int,
    concepts: List[str],
    instructions: List[str],
    batch_size: int = 8,
    judge_lm="gpt-4o-mini",
    temperature=0.01,
    max_new_tokens=300,
):
    """
    Batch run evaluation with API judge LLM.
    """
    assert len(concepts) == len(instructions)

    lm = RemoteAPIModel(
        model=judge_lm,
        client=client,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    n_batches = (len(concepts) + batch_size - 1) // batch_size
    results = defaultdict(list)

    for batch_idx in tqdm(range(n_batches), desc=f"Generating steering prompt for concept [{concept_id}]"):
        batch_prompts = []
        for i in range(batch_size):
            concept = concepts[batch_idx*batch_size + i]
            inst = instructions[batch_idx*batch_size + i]
            batch_prompts.append(
                T_GENERATE_STEERING_PROMPT_BLEND.format(
                    concept=concept, instruction=inst
                )
            )

            results["original_prompt"].append(inst)
            results["concept_id"].append(concept_id)

        judge_results = asyncio.run(
            lm.chat_completions(prompts=batch_prompts, batch_size=batch_size)
        )

        for i, resp in enumerate(judge_results):
            results['steered_prompt'].append(resp.strip())
    return results


def steering_prompt_prepend_async(
    client: AsyncClient,
    concepts: List[str],
    batch_size: int = 8,
    judge_lm="gpt-4o-mini",
    temperature=0.01,
    max_new_tokens=300,
):
    """
    Batch run evaluation with API judge LLM.
    """

    lm = RemoteAPIModel(
        model=judge_lm,
        client=client,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    n_batches = (len(concepts) + batch_size - 1) // batch_size
    results = defaultdict(list)

    for batch_idx in tqdm(range(n_batches), desc="Generating steering prompts"):
        batch_prompts = []
        for i in range(batch_size):
            concept = concepts[batch_idx*batch_size + i]
            batch_prompts.append(
                T_GENERATE_PREPEND_STEERING_PROMPT.format(
                    concept=concept,
                )
            )

        judge_results = asyncio.run(
            lm.chat_completions(prompts=batch_prompts, batch_size=batch_size)
        )

        for i, resp in enumerate(judge_results):
            results['steering_prompt'].append(resp.strip())
        results['concept'] = concepts
    return results


def suppression_prompt_blended_async(
    client: AsyncClient,
    concept_id: int,
    concepts: List[str],
    instructions: List[str],
    batch_size: int = 8,
    judge_lm="gpt-4o-mini",
    temperature=0.01,
    max_new_tokens=300,
):
    """
    Batch run evaluation with API judge LLM.
    """
    assert len(concepts) == len(instructions)

    lm = RemoteAPIModel(
        model=judge_lm,
        client=client,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    n_batches = (len(concepts) + batch_size - 1) // batch_size
    results = defaultdict(list)

    for batch_idx in tqdm(range(n_batches), desc=f"Generating suppression prompt for concept [{concept_id}]"):
        batch_prompts = []
        for i in range(batch_size):
            concept = concepts[batch_idx*batch_size + i]
            inst = instructions[batch_idx*batch_size + i]
            batch_prompts.append(
                T_GENERATE_SUPPRESSION_PROMPT_BLEND.format(
                    concept=concept, instruction=inst
                )
            )

            results["original_prompt"].append(inst)
            results["concept_id"].append(concept_id)

        judge_results = asyncio.run(
            lm.chat_completions(prompts=batch_prompts, batch_size=batch_size)
        )

        for i, resp in enumerate(judge_results):
            results['steered_prompt'].append(resp.strip())
    return results
