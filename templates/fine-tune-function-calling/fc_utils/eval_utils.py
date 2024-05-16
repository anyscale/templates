from typing import Union, Tuple, List
from tqdm import tqdm
from enum import Enum
import ray.data

from fc_utils.response_parsers import (
    ERROR_OUTPUT,
    AnyscaleResponseParser,
    OpenAIResponseParser,
    INCORRECT_FORMAT
)


class PossibleMistakes(Enum):
    UNWANTED_FUNCTION_CALL = "Unwanted Function Call"
    NO_FUNCTION_CALL = "No Function Call"
    INCORRECT_FORMAT = "Incorrect Function Format"
    INCORRECT_NUMBER_OF_FUNCTION_CALLS = "Incorrect Number of Function Calls"
    WRONG_FUNCTION_NAME = "Wrong Function Name"
    WRONG_ARGUMENT_VALUE = "Wrong Argument Value"
    MISSING_ARGUMENT = "Missing Argument"

    def values():
        return [item.value for item in PossibleMistakes]

def is_match(response: dict, ground_truth: dict) -> Tuple[bool, str]:
    """
    Checks if the response matches the ground truth. Returns a boolean and a string indicating the mistake type if any.
    """
    if ground_truth["tool_calls"] is None:
        if response["tool_calls"] is None:
            return True, ""
        else:  # explicit else for clarity
            return False, PossibleMistakes.UNWANTED_FUNCTION_CALL

    if response["tool_calls"] is None:
        return False, PossibleMistakes.NO_FUNCTION_CALL
    elif response["tool_calls"] == INCORRECT_FORMAT: # error during parsing
        return False, PossibleMistakes.INCORRECT_FORMAT
    elif len(response["tool_calls"]) != len(ground_truth["tool_calls"]):
        return False, PossibleMistakes.INCORRECT_NUMBER_OF_FUNCTION_CALLS

    for expected_tool_call, actual_tool_call in zip(ground_truth["tool_calls"], response["tool_calls"]):
        if "name" not in actual_tool_call or "arguments" not in actual_tool_call:
            return False, PossibleMistakes.INCORRECT_FORMAT
        if expected_tool_call["name"] != actual_tool_call["name"]:
            return False, PossibleMistakes.WRONG_FUNCTION_NAME
        elif expected_tool_call["arguments"] != actual_tool_call["arguments"]:
            if len(expected_tool_call["arguments"]) != len(actual_tool_call["arguments"]):
                return False, PossibleMistakes.MISSING_ARGUMENT
            return False, PossibleMistakes.WRONG_ARGUMENT_VALUE
    return True, ""


def parse_and_eval(
    parser: Union[OpenAIResponseParser, AnyscaleResponseParser],
    messages,
    expected_responses,
    tools=None,
):
    """
    Parse and eval loop to parse the assistant responses and evaluate them against the ground truth in the conversation.
    Args:
    parser: OpenAIResponseParser or AnyscaleResponseParser object
    messages: list of messages in the conversation. The messages with role 'assistant' are ignored and triggers a chat completion call
    expected_responses: list of ground truth responses
    tools: list of tools to available, used by OpenAIResponseParser
    """
    assert messages[0]["role"] == "system", "First message must be from system"
    match = True
    current_conv = []
    assist_id = 0
    conv_id = 0
    while conv_id < len(messages):
        if messages[conv_id]["role"] != "assistant":
            current_conv.append(messages[conv_id])
        # 'role' is 'assistant'. Thus, get assistant response
        else:
            if current_conv[-1]["role"] == "tool":
                # If the last message was a tool, we need additional processing.
                # For a tool, we need to have a valid tool call id. This can only be retrieved from the previous assistant response
                # because our dataset is synthetically constructed
                if (
                    current_conv[-2]["role"] == "assistant"
                    and current_conv[-2]["tool_calls"]
                ):
                    current_conv[-1]["tool_call_id"] = current_conv[-2]["tool_calls"][0].id
                else:
                    return current_conv, False
            if isinstance(parser, OpenAIResponseParser):
                parsed_response = parser.get_parsed_response(current_conv, tools)
            else:
                parsed_response = parser.get_parsed_response(current_conv)
            if parsed_response["content"] == ERROR_OUTPUT:
                return None, None  # return None if there's an error

            is_correct, mistake_type = is_match(parsed_response, expected_responses[assist_id])
            match = match and (is_correct)
            original_assistant_response = parsed_response["original_response"]
            current_conv.append(dict(original_assistant_response))
            assist_id += 1 # next expected response
            if not match:  # return right away if model output is incorrect
                return current_conv, match, mistake_type
        conv_id += 1 # next message
    return current_conv, match, mistake_type


def evaluate_gpt(ds: List[dict], openai_parser: OpenAIResponseParser):
    """
    Evaluates the GPT4 model on the test dataset.
    """
    num_correct = 0
    total_count = 0
    results = []
    pbar = tqdm(total=len(ds), desc="Evaluating GPT4..")
    for example in ds:
        openai_messages = example["openai_messages"]
        openai_tools = [
            {"type": "function", "function": fn} for fn in example["tools"]
        ]
        openai_conv, openai_is_match, mistake_type = parse_and_eval(
            openai_parser, openai_messages, example["expected_responses"], tools=openai_tools
        )
        pbar.update(1)
        if openai_conv is None:  # skip if api errors out
            openai_is_match = None
        else:
            total_count += 1
        result_dict = {"correct": openai_is_match, "mistake_type": mistake_type, "example": example, "conv": openai_conv}
        results.append(result_dict)

        if openai_is_match:
            num_correct += 1
    print("GPT4 accuracy: ", num_correct / total_count)
    return results

def evaluate_finetuned(ds: List[dict], anyscale_parser: AnyscaleResponseParser):
    """
    Evaluates our fine-tuned model on the test dataset.
    """
    anyscale_accuracy = 0
    total_count = 0
    results = []
    pbar = tqdm(total=len(ds), desc="Evaluating Finetuned Model..")
    for example in ds:
        anyscale_messages = example["anyscale_messages"]
        anyscale_conv, anyscale_is_match, mistake_type = parse_and_eval(anyscale_parser, anyscale_messages, example["expected_responses"])
        pbar.update(1)
        if anyscale_conv is None: # skip if api errors out
            anyscale_is_match = None
        else:
            total_count += 1
        if anyscale_is_match:
            anyscale_accuracy += 1
        result_dict = {"correct": anyscale_is_match, "mistake_type": mistake_type, "example": example, "conv": anyscale_conv}

        results.append(result_dict)
    print("Finetuned model accuracy: ", anyscale_accuracy / total_count)
    return results
