"""
Utilities for synthetic data generation
"""
import logging
import os
import random
import re
import string
import time
import unicodedata
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import openai
import requests
from huggingface_hub import repo_exists, snapshot_download
from openai import OpenAI
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from src.utils.models import OfflineInferenceConfig, OnlineInferenceConfig
from src.utils.common import get_completion, init_logger

logger = init_logger()

# TODO: See if this parameter can be removed entirely
VLLM_MAX_MODEL_LEN = 8192

class InferenceType(Enum):
    ONLINE = "online"
    OFFLINE = "offline"


def format_into_prompt(
    row: Dict[str, Any],
    template: str,
    col_name: str,
    type: InferenceType,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
):
    """Given a prompt template, formats the keys from the Dataset row based on the inference type

    If `type` is offline, then the input is formatted into raw text by applying the tokenizer's default chat template
    Else, the input is formatted into the OpenAI messages format
    """
    if type == InferenceType.OFFLINE:
        return format_into_prompt_rawtext(
            row=row, template=template, col_name=col_name, tokenizer=tokenizer
        )
    else:
        return format_into_prompt_openai(row=row, template=template, col_name=col_name)


def format_into_prompt_openai(row: Dict[str, Any], template: str, col_name: str):
    """Given a prompt template, format the keys from the Dataset row into the OpenAI messages format"""
    row[col_name] = [{"content": template.format(**row), "role": "user"}]
    return row


def format_into_prompt_rawtext(
    row: Dict[str, Any],
    template: str,
    col_name: str,
    tokenizer: PreTrainedTokenizerBase,
):
    """Given a prompt template, format the keys from the Dataset row into the template as plaintext using a tokenizer's chat template"""
    row[col_name] = tokenizer.apply_chat_template(
        [{"content": template.format(**row), "role": "user"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return row


def duplicate_rows(row, count, id_col):
    return [{**row, id_col: i} for i in range(count)]


# TODO: clean up
def process_question(text, num_questions=5, letter_choices=("A", "B", "C", "D", "E")):
    questions = []
    assert all(
        len(choice) == 1 for choice in letter_choices
    ), f"Letter choices must be single letters, got {letter_choices}"
    separator = ". "
    choice_prefixes = [
        f"{letter_choice}{separator}" for letter_choice in letter_choices
    ]
    prefix_length = len(choice_prefixes[0])
    for line in text.split("\n"):
        if re.match(r"Q\d\) ", line[:4]):
            questions.append({"question": line.split(") ")[1], "choices": {}})
        elif any([line.lstrip().startswith(prefix) for prefix in choice_prefixes]):
            questions[-1]["choices"][line[0]] = line.lstrip()[prefix_length:]
        elif "Answer: " in line:
            questions[-1]["answer"] = line.split("Answer: ")[-1][0]

    assert len(questions) == num_questions
    for question in questions:
        assert len(question["choices"]) == len(letter_choices)
        assert "answer" in question and question["answer"] in letter_choices
    return questions


# TODO: clean up
def write_questions(questions, letter_choices=("A", "B", "C", "D", "E")):
    random.shuffle(questions)
    prompt = ""
    answers = []
    for i, question in enumerate(questions):
        prompt += f"Q{i + 1}) " + question["question"] + "\n"
        options = list(question["choices"].items())
        random.shuffle(options)
        for j in range(len(letter_choices)):

            orig_letter, choice_text = options[j]

            if orig_letter == question["answer"]:
                answers.append(letter_choices[j])

            prompt += f"{letter_choices[j]}. " + choice_text + "\n"
        prompt += "\n"
    return prompt, np.array(answers)


def shuffle_qa(row, col_in, col_out_prompt, col_out_answers):
    try:
        row[col_out_prompt], row[col_out_answers] = write_questions(
            process_question(row[col_in])
        )
        return [row]
    except Exception as e:
        return []


def extract_answers(row, col_in, col_out, num_questions):
    text = row[col_in]
    if text is None:
        row[col_out] = ["No Judge Output"]
        return row
    answers = ["Unsure"] * num_questions
    for line in text.split("\n"):
        if m := re.match(r"Q(\d)\) ([A-E])", line.strip()):
            if 1 <= int(m.group(1)) <= num_questions:
                answers[int(m.group(1)) - 1] = m.group(2)

    row[col_out] = answers
    return row


class OfflinePredictor:
    def __init__(
        self,
        model_config: OfflineInferenceConfig,
        col_in: str,
        col_out: str,
    ):
        logger = init_logger()
        self.col_in = col_in
        self.col_out = col_out
        self.lora_location = model_config.adapter_id_or_path
        self.vllm_settings = dict(
            tensor_parallel_size=model_config.scaling_config.num_gpus_per_instance,
            max_model_len=VLLM_MAX_MODEL_LEN,
            dtype="bfloat16",
        )

        # Create a sampling params object.
        self.sampling_params = SamplingParams(
            n=1,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            top_p=model_config.top_p,
            stop=["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"],
        )

        llm_args = dict(model=model_config.model_id_or_path)

        if self.lora_location is not None:
            if repo_exists(self.lora_location):
                self.lora_location = snapshot_download(self.lora_location)
                logger.info("Downloading LoRA to:", self.lora_location)
            llm_args.update(
                dict(
                    enable_lora=True,
                    max_loras=1,
                    max_lora_rank=64,
                )
            )

        if self.vllm_settings is not None:
            llm_args.update(self.vllm_settings)

        # Create an LLM.
        self.llm = LLM(**llm_args)

    def __call__(self, batch):
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        if self.lora_location is not None:
            outputs = self.llm.generate(
                list(batch[self.col_in]),
                self.sampling_params,
                lora_request=LoRARequest("lora", 1, self.lora_location),
            )
        else:
            outputs = self.llm.generate(list(batch[self.col_in]), self.sampling_params)

        generated_text = []
        for i, output in enumerate(outputs):
            generated_text.append(output.outputs[0].text.strip())

        return {
            **batch,
            self.col_out: generated_text,
        }


class OnlinePredictor:
    def __init__(
        self,
        model_config: OnlineInferenceConfig,
        col_in: str,
        col_out: str,
    ):
        if not os.environ.get(model_config.api_key_env_var):
            raise ValueError(
                f"API Key must be set through {model_config.api_key_env_var}"
            )
        self.client = OpenAI(
            base_url=model_config.base_url,
            api_key=os.environ[model_config.api_key_env_var],
        )
        self.model = model_config.model_id
        self.col_in = col_in
        self.col_out = col_out
        self.temperature = model_config.temperature
        self.max_tokens = model_config.max_tokens

    def __call__(self, example):
        try:
            resp = get_completion(
                client=self.client,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=list(example[self.col_in]),
            )
            example[self.col_out] = resp.choices[0].message.content
        except Exception as e:
            logger.error(
                f"Error generating response:  {e} for input {example[self.col_in]}"
            )
            example[self.col_out] = None
        return example



def get_predictions_on_dataset(ds, model_config: Union[OnlineInferenceConfig, OfflineInferenceConfig], col_in: str, col_out: str):
    if isinstance(model_config, OfflineInferenceConfig):
        ds = ds.map_batches(
            OfflinePredictor,
            fn_constructor_kwargs=dict(
                col_in=col_in,
                col_out=col_out,
                model_config=model_config,
            ),
            num_gpus=model_config.scaling_config.num_gpus_per_instance,
            concurrency=model_config.scaling_config.concurrency,
            batch_size=model_config.scaling_config.batch_size,
            resources=model_config.scaling_config.custom_resources,
            zero_copy_batch=True,
            batch_format="numpy",
        )
    else:
        ds = ds.map(
            OnlinePredictor,
            fn_constructor_kwargs=dict(
                col_in=col_in,
                col_out=col_out,
                model_config=model_config,
            ),
            concurrency=model_config.concurrency,
        )
    return ds
