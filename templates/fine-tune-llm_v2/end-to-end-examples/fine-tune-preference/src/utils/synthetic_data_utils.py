"""
Utilities for synthetic data generation
"""

import json
import logging
import os
import random
import re
import time
import unicodedata
from enum import Enum
from operator import is_
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import openai
import requests
from huggingface_hub import repo_exists, snapshot_download
from openai import OpenAI
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from src.utils.common import get_completion, init_logger
from src.utils.download import download_to_local, is_remote_path
from src.utils.models import OfflineInferenceConfig, OnlineInferenceConfig

if TYPE_CHECKING:
    from ray.data import Dataset

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


def duplicate_rows(
    row: Dict[str, Any], count: int, id_col: str
) -> List[Dict[str, Any]]:
    """Duplicates a row for the specified number of times.

    Adds an additional column `id_col` to each duplicated row, containing a unique index.
    Args:
        row: A dict representing a row in the dataframe
        count: The number of times to replicate a row
        id_col: Column name for the new column with the duplication index.
    Returns:
        The list of duplicated rows
    """
    return [{**row, id_col: i} for i in range(count)]


def process_question(
    text: str,
    num_questions: int = 5,
    letter_choices: Tuple[str, ...] = ("A", "B", "C", "D", "E"),
) -> List[Dict[str, Any]]:
    """Parses raw text containing questions, options and answers into a list of dicionaries.

    Args:
        text: Raw string containing a list of questions and multiple choice options.
        num_questions: Number of questions in the text
        letter_choices: The list of letter choices for each question.
    Returns:
        questions: A list of dictionaries with keys "question", "answer" and "choices"
    """
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


def extract_answers(row: Dict[str, Any], col_in: str, col_out: str, num_questions: int):
    """Extracts answers from the raw text column in the row"""
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


def dump_jsonl_to_string(row: Dict[str, Any], col: str) -> Dict[str, Any]:
    """Converts the given column in the row in jsonl format to a string.

    Useful to avoid serialization issues with pyarrow
    """
    row[col] = json.dumps(list(row[col]))
    return row


def download_model(path: str):
    """Helper function to download a model given the path"""
    if not is_remote_path(path):
        return path
    return download_to_local(path)


class OfflinePredictor:
    def __init__(
        self,
        model_config: OfflineInferenceConfig,
        col_in: str,
        col_out: str,
    ):
        logger = init_logger()

        model_path = download_model(model_config.model_id_or_path)
        adapter_path = None
        if model_config.adapter_id_or_path:
            adapter_path = download_model(model_config.adapter_id_or_path)

        self.col_in = col_in
        self.col_out = col_out
        self.lora_location = adapter_path
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

        llm_args = dict(model=model_path)

        if self.lora_location is not None:
            # at this stage, lora_location should contain a local path or hf repo id
            if not os.path.exists(self.lora_location):
                repo_exists(self.lora_location)  # make sure it exists
                logger.info("Downloading LoRA to:", self.lora_location)
                self.lora_location = snapshot_download(self.lora_location)
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


def get_predictions_on_dataset(
    ds: "Dataset",
    model_config: Union[OnlineInferenceConfig, OfflineInferenceConfig],
    col_in: str,
    col_out: str,
):
    """Get predictions for a model on the given dataset using Ray data

    Supports online/offline inference given the model config.
    Args:
        ds: The input dataset
        model_config: Model inference config. Can be online/ offline.
        col_in: Input column in the dataset.
        col_out: Output column to write the results to.
    """
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
            accelerator_type=model_config.scaling_config.accelerator_type,
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
