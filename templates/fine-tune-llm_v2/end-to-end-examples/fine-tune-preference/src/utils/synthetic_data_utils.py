"""
Utilities for synthetic data generation
"""

import json
import random
import re
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from transformers import PreTrainedTokenizerBase

from src.utils.common import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger()

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
    except Exception:
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
