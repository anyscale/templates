"""
Evaluation utilities
"""

from typing import Union, Tuple, List, Dict, Any
from tqdm import tqdm
from enum import Enum
from dataclasses import dataclass

from fc_utils.response_parsers import ERROR_OUTPUT, ResponseParser, INCORRECT_FORMAT, ParsedResponse
from fc_utils.data_format import MessageType, ToolCallType, check_tool_call_format


class Mistakes(Enum):
    """Enum for different mistakes in the model's response."""

    UNWANTED_FUNCTION_CALL = "Unwanted Function Call"
    NO_FUNCTION_CALL = "No Function Call"
    INCORRECT_FORMAT = "Incorrect Function Format"
    INCORRECT_NUMBER_OF_FUNCTION_CALLS = "Incorrect Number of Function Calls"
    WRONG_FUNCTION_NAME = "Wrong Function Name"
    WRONG_ARGUMENT_VALUE = "Wrong Argument Value"
    MISSING_ARGUMENT = "Missing Argument"
    NONE = ""

    @classmethod
    def values(cls):
        """Returns a list of all mistake values."""
        return [item.value for item in Mistakes]


class Model(Enum):
    """Enum for different models during evaluation."""

    GPT = "gpt"
    FINETUNED = "finetuned"


@dataclass
class Result:
    """Dataclass to store the evaluation results."""
    is_correct: bool
    is_valid: bool
    mistake_type: Mistakes
    generated_conv: List[MessageType]
    ground_truth_conv: List[MessageType]


def check_match(
    response: ParsedResponse,
    ground_truth: Dict[str, Union[str, ToolCallType]],
) -> Tuple[bool, Mistakes]:
    """Checks if the response matches the ground truth. Returns a boolean and a string indicating the mistake type if any."""
    is_match = True
    reason = Mistakes.NONE
    response_tool_calls = response.tool_calls
    ground_truth_tool_calls = ground_truth["tool_calls"]
    # If the ground truth has no tool calls, the response should not have any tool calls
    if ground_truth_tool_calls is None:
        if response_tool_calls is not None:
            is_match = False
            reason = Mistakes.UNWANTED_FUNCTION_CALL
    # If the ground truth has tool calls, the response has to make a tool call
    elif response_tool_calls is None:
        is_match = False
        reason = Mistakes.NO_FUNCTION_CALL
    # If the ground truth has tool calls, then the response tool call has to be in the correct format
    elif response_tool_calls == INCORRECT_FORMAT or not check_tool_call_format(
        response_tool_calls
    ):
        is_match = False
        reason = Mistakes.INCORRECT_FORMAT
    # If the ground truth has tool calls, then the response should have the same number of tool calls
    elif len(response_tool_calls) != len(ground_truth_tool_calls):
        is_match = False
        reason = Mistakes.INCORRECT_NUMBER_OF_FUNCTION_CALLS
    else:
        is_match, reason = compare_tool_calls(response_tool_calls, ground_truth_tool_calls)
    return is_match, reason


def compare_tool_calls(response_tool_calls: List[ToolCallType], ground_truth_tool_calls: List[ToolCallType]):
    # Sort list of tool calls by function name
    sorted_gt_tool_calls = sorted(
        ground_truth_tool_calls, key=lambda x: x["function"]["name"]
    )
    sorted_response_tool_calls = sorted(
        response_tool_calls, key=lambda x: x["function"]["name"]
    )
    is_match = True
    reason = Mistakes.NONE
    for expected_tool_call, actual_tool_call in zip(
        sorted_gt_tool_calls, sorted_response_tool_calls
    ):
        expected_function = expected_tool_call["function"]
        actual_function = actual_tool_call["function"]
        # If the ground truth has a tool call, then the response tool call has to have the same name and arguments
        # If either field doesn't match, return immediately
        if expected_function["name"] != actual_function["name"]:
            is_match = False
            reason = Mistakes.WRONG_FUNCTION_NAME
            return is_match, reason
        elif expected_function["arguments"] != actual_function["arguments"]:
            is_match = False
            if len(expected_function["arguments"]) != len(
                actual_function["arguments"]
            ):
                reason = Mistakes.MISSING_ARGUMENT
            else:
                reason = Mistakes.WRONG_ARGUMENT_VALUE
            return is_match, reason
    return is_match, reason


def parse_and_eval(
    parser: ResponseParser, example: Dict[str, Any]
) -> Tuple[List[Dict[str, str]], bool, Mistakes]:
    """
    Parse and eval loop to parse the assistant responses and evaluate them against the ground truth in the conversation.

    The input example consists of the ground truth conversation. The function iterates through the messages and when it encounters a message with role 'assistant' - it calls the parser to get the model's response.

    Args:
        parser: OpenAIResponseParser or AnyscaleResponseParser object
        example: Example to evaluate
    Returns:
        generated_conv: The full list of messages with model generated responses
        match: Boolean indicating if the generated conversation matches the ground truth
        mistake_type: Mistake enum indicating the type of mistake made by the model if any
    """
    messages = example["messages"]
    # Use safe indexing since this entry is optional
    tools = example.get("tools", None)
    is_match = True
    generated_conv = []
    for message in messages:
        if message["role"] != "assistant":
            generated_conv.append(message)
        else:
            # Get the model's response
            parsed_response = parser.get_parsed_response(generated_conv, tools)

            if parsed_response.content == ERROR_OUTPUT:
                # return None if there's an error
                return None, None, None

            # Evaluate against the ground truth/ the current message
            _match, mistake_type = check_match(parsed_response, message)
            is_match = is_match and _match
            original_assistant_response = parsed_response.original_response
            # Convert response object to dict and append to the current conversation
            generated_conv.append(dict(original_assistant_response))
            # return right away if model output is incorrect
            if not is_match:
                return generated_conv, is_match, mistake_type
    return generated_conv, is_match, mistake_type


def evaluate_model(
    dataset: List[Dict[str, Any]], parser: ResponseParser, model: Model
) -> Tuple[List[Dict[str, Any]], float]:
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
        model: Model enum indicating the model type to evaluate.

    Returns:
        results: List of results
        accuracy: Float indicating the accuracy of the model
    """
    # Initialize lists to store results
    results = []
    # Stores list of matches for valid results
    corrects = []
    pbar_desc = (
        "Evaluating GPT4..." if model == Model.GPT else "Evaluating Finetuned Model..."
    )
    pbar = tqdm(total=len(dataset), desc=pbar_desc)
    messages = example["messages"]
    # Use safe indexing since this entry is optional
    tools = example.get("tools", None)
    for example in dataset:
        # entry is valid by default
        is_valid = False

        # query the model, parse and evaluate the generated responses
        conv, is_correct, mistake_type = parse_and_eval(parser, example)

        is_valid = conv is not None

        if is_valid:
            corrects.append(is_correct)

        result = Result(
            is_correct=is_correct,
            is_valid=is_valid,
            mistake_type=mistake_type,
            generated_conv=conv,
            ground_truth_conv=messages,
        )
        results.append(result)
        pbar.update(1)
    pbar.close()

    accuracy = sum(corrects) / len(corrects) if corrects else 0.0
    return results, accuracy
