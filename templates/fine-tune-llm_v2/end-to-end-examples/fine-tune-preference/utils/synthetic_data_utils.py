import logging
import random
import re
from typing import Any, Dict

import numpy as np
import requests
from huggingface_hub import repo_exists, snapshot_download
from transformers import PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from utils.utils import init_logger


def format_into_prompt_openai(
    row: Dict[str, Any], template: str, settings: Dict[str, Any], col_name: str
):
    """
    Given a prompt template, format the keys from the Dataset row into the template as an OpenAI-style request
    """
    row[col_name] = dict(
        **settings, messages=[{"content": template.format(**row), "role": "user"}]
    )
    return row


def format_into_prompt_rawtext(
    row: Dict[str, Any],
    template: str,
    tokenizer: PreTrainedTokenizerBase,
    col_name: str,
):
    """
    Given a prompt template, format the keys from the Dataset row into the template as plaintext using a tokenizer's chat template
    """
    row[col_name] = tokenizer.apply_chat_template(
        [{"content": template.format(**row), "role": "user"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return row

def duplicate_rows(row, count, id_col):
    return [
        {**row, id_col: i} for i in range(count)
    ]

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
    except Exception as e:
        row[col_out_prompt], row[col_out_answers] = None, None
    return row

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

class LLMPredictor:
    def __init__(
        self,
        model_location,
        col_in,
        col_out,
        temperature,
        max_tokens,
        lora_location=None,
        vllm_settings=None,
    ):
        logger = init_logger()
        self.col_in = col_in
        self.col_out = col_out
        self.lora_location = lora_location

        # Create a sampling params object.
        self.sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["<|eot_id|>", "<|end_of_text|>"],
        )

        llm_args = dict(model=model_location)

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

        if vllm_settings is not None:
            llm_args.update(vllm_settings)

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
