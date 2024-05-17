"""
Evaluation utilities
"""

from typing import Union, Tuple, List, Dict, Any
from tqdm import tqdm
from enum import Enum

from fc_utils.response_parsers import (
    ERROR_OUTPUT,
    ResponseParser,
    INCORRECT_FORMAT
)
from fc_utils.function_extraction_utils import ToolCallType

class Mistakes(Enum):
    UNWANTED_FUNCTION_CALL = "Unwanted Function Call"
    NO_FUNCTION_CALL = "No Function Call"
    INCORRECT_FORMAT = "Incorrect Function Format"
    INCORRECT_NUMBER_OF_FUNCTION_CALLS = "Incorrect Number of Function Calls"
    WRONG_FUNCTION_NAME = "Wrong Function Name"
    WRONG_ARGUMENT_VALUE = "Wrong Argument Value"
    MISSING_ARGUMENT = "Missing Argument"
    NONE = ""

    def values():
        return [item.value for item in Mistakes]

def check_match(response: Dict[str, Union[str, ToolCallType]], ground_truth: Dict[str, Union[str, ToolCallType]]) -> Tuple[bool, Mistakes]:
    """
    Checks if the response matches the ground truth. Returns a boolean and a string indicating the mistake type if any.
    """
    is_match = True
    reason = Mistakes.NONE

    # if the ground truth has no tool calls, the response should not have any tool calls
    if ground_truth["tool_calls"] is None:
        if response["tool_calls"] is not None:
            is_match =  False
            reason = Mistakes.UNWANTED_FUNCTION_CALL
    else:
        # if the ground truth has tool calls, the response has to make a tool call
        if response["tool_calls"] is None:
            is_match =  False
            reason = Mistakes.NO_FUNCTION_CALL
        # if the ground truth has tool calls, then the response tool call has to be in the correct format
        elif response["tool_calls"] == INCORRECT_FORMAT or not check_format(response["tool_calls"]):
            is_match = False
            reason = Mistakes.INCORRECT_FORMAT
        # if the ground truth has tool calls, then the response should have the same number of tool calls
        elif len(response["tool_calls"]) != len(ground_truth["tool_calls"]):
            is_match = False
            reason = Mistakes.INCORRECT_NUMBER_OF_FUNCTION_CALLS
        else:
            # sort list of tool calls by function name
            sorted_gt_tool_calls = sorted(ground_truth["tool_calls"], key=lambda x: x["name"])
            sorted_response_tool_calls = sorted(response["tool_calls"], key=lambda x: x["name"])
            for expected_tool_call, actual_tool_call in zip(sorted_gt_tool_calls, sorted_response_tool_calls):
                # if the ground truth has a tool call, then the response tool call has to have the same name and arguments
                if expected_tool_call["name"] != actual_tool_call["name"]:
                    is_match = False
                    reason = Mistakes.WRONG_FUNCTION_NAME
                elif expected_tool_call["arguments"] != actual_tool_call["arguments"]:
                    is_match = False
                    if len(expected_tool_call["arguments"]) != len(actual_tool_call["arguments"]):
                        reason = Mistakes.MISSING_ARGUMENT
                    else:
                        reason = Mistakes.WRONG_ARGUMENT_VALUE
    return is_match, reason


def check_format(tool_calls: List[ToolCallType]) -> bool:
    """
    Checks if the tool call is in the correct format.
    """
    for tool_call in tool_calls:
        if "name" not in tool_call or "arguments" not in tool_call or not isinstance(tool_call["arguments"], dict) or not isinstance(tool_call["name"], str):
            return False
    return True


def parse_and_eval(
    parser: ResponseParser,
    messages: List[Dict[str, str]],
    expected_responses: List[Dict[str, Union[str, ToolCallType]]],
    tools: List[Dict[str, Any]]=None,
) -> Tuple[List[Dict[str, str]], bool, Mistakes]:
    """
    Parse and eval loop to parse the assistant responses and evaluate them against the ground truth in the conversation.

    The function iterates through the messages in the conversation. When it encounters a message with role 'assistant' - this is the raw expected response -  it calls the parser to get the model's actual response.

    Args:
        parser: OpenAIResponseParser or AnyscaleResponseParser object
        messages: List of messages in the conversation. The messages with role 'assistant' are ignored and triggers a chat completion call
        expected_responses: List of parsed ground truth responses. This can be just one response (single-turn) or multiple responses (multi-turn).
        tools: List of tools to available, used by OpenAIResponseParser

    Returns:
        current_conv: The full list of messages with model generated responses
        match: Boolean indicating if the generated conversation matches the ground truth
    """
    assert messages[0]["role"] == "system", "First message must be from system"
    is_match = True
    current_conv = []
    # track the previous assistant response
    previous_response = None
    assist_idx = 0
    conv_idx = 0
    while conv_idx < len(messages):
        if messages[conv_idx]["role"] != "assistant":
            current_conv.append(messages[conv_idx])
        # 'role' is 'assistant'. Thus, get assistant response
        else:
            # If the last message was a tool, we need additional processing.
            if current_conv[-1]["role"] == "tool":
                # For a tool, we need to have a valid tool call id. This can only be retrieved from the previous assistant response
                # because our dataset is synthetically constructed
                if previous_response["tool_calls"] is not None:
                    current_conv[-1]["tool_call_id"] = current_conv[-2]["tool_calls"][0].id
                else:
                    # previous response didn't have a tool call, but we expected one. Return with match=False
                    return current_conv, False, Mistakes.NO_FUNCTION_CALL

            parsed_response = parser.get_parsed_response(current_conv, tools)

            if parsed_response["content"] == ERROR_OUTPUT:
                # return None if there's an error
                return None, None, None

            _match, mistake_type = check_match(parsed_response, expected_responses[assist_idx])
            is_match = is_match and _match
            original_assistant_response = parsed_response["original_response"]
            current_conv.append(dict(original_assistant_response))

            # return right away if model output is incorrect
            if not is_match:
                return current_conv, is_match, mistake_type

            previous_response = parsed_response
            # next expected response
            assist_idx += 1
        # next message
        conv_idx += 1
    return current_conv, is_match, mistake_type


def evaluate_model(dataset: List[Dict[str, Any]], parser: ResponseParser, model: str) -> Tuple[List[Dict[str, Any]], float]:
    """
    Evaluates the given model on the test dataset. The function returns a list of results and the accuracy.

    Each result entry is a dictionary containing the following:
        - correct: Boolean indicating if the model's response matches the ground truth
        - invalid_entry: Boolean indicating if the entry is invalid. This can happen if the api call failed for some reason (max retries reached, etc.). Such entries are skipped in accuracy calculation.
        - mistake_type: String indicating the type of mistake made by the model if any
        - example: The original example from the dataset
        - conv: The full conversation with the model's actual response.

    Args:
        dataset: List of examples to evaluate
        openai_parser: ResponseParser object

    Returns:
        results: List of results
    """
    if model not in ["gpt", "finetuned"]:
        raise ValueError(f"Model must be either 'gpt' or 'finetuned', got {model}")
    num_correct = 0
    total_count = 0
    results = []
    pbar_desc = "Evaluating GPT4..." if model == "gpt" else "Evaluating Finetuned Model..."
    pbar = tqdm(total=len(dataset), desc=pbar_desc)

    for example in dataset:
        if model == "gpt":
            messages = example["openai_messages"]
            tools = [
                {"type": "function", "function": fn} for fn in example["tools"]
            ]
        else:
            messages = example["anyscale_messages"]
            tools = None

        is_match = False
        # entry is valid by default
        invalid_entry = False

        # query the model, parse and evaluate the generated responses
        conv, is_match, mistake_type = parse_and_eval(
            parser, messages, example["expected_responses"], tools=tools
        )

        # skip if api errors out
        if conv is None:
            invalid_entry = True
        else:
            # valid, increment total count
            total_count += 1

            if is_match:
                num_correct += 1

        result_dict = {"correct": is_match, "invalid_entry": invalid_entry,  "mistake_type": mistake_type, "example": example, "conv": conv}
        results.append(result_dict)
        pbar.update(1)

    return results, num_correct / total_count
