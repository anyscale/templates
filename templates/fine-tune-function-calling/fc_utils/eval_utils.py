from typing import Union

from fc_utils.response_parsers import (
    ERROR_OUTPUT,
    AnyscaleResponseParser,
    OpenAIResponseParser,
    INCORRECT_FORMAT
)
from tqdm import tqdm

POSSIBLE_MISTAKES = ["Unwanted Function Call", "No Function Call", "Incorrect Function Format", "Incorrect Number of Function Calls" ,"Wrong Function Name", "Wrong Argument Value", "Missing Argument"]


def is_match(response: dict, ground_truth: dict):
    """
    Checks if the response matches the ground truth.
    """
    if ground_truth["tool_calls"] is None:
        if response["tool_calls"] is None:
            return True, ""
        else:  # explicit else for clarity
            return False, POSSIBLE_MISTAKES[0]

    if response["tool_calls"] is None:
        return False, POSSIBLE_MISTAKES[1]
    if response["tool_calls"] == INCORRECT_FORMAT:
        return False, POSSIBLE_MISTAKES[2]
    if len(response["tool_calls"]) != len(ground_truth["tool_calls"]):
        return False, POSSIBLE_MISTAKES[3]
    for expected_tool_call, actual_tool_call in zip(ground_truth["tool_calls"], response["tool_calls"]):
        if "name" not in actual_tool_call or "arguments" not in actual_tool_call:
            return False, POSSIBLE_MISTAKES[2] # incorrect format
        if expected_tool_call["name"] != actual_tool_call["name"]:
            return False, POSSIBLE_MISTAKES[4]
        elif expected_tool_call["arguments"] != actual_tool_call["arguments"]:
            if len(expected_tool_call["arguments"]) != len(actual_tool_call["arguments"]):
                return False, POSSIBLE_MISTAKES[6]
            return False, POSSIBLE_MISTAKES[5]
    return True, ""


def parse_and_eval(
    parser: Union[OpenAIResponseParser, AnyscaleResponseParser],
    messages,
    expected_responses,
    tools=None,
):
    """
    Parse and eval loop to parse the assistant responses and evaluate them against the ground truth in the conversation.
    This assumes that an assistant response is expected after every user/tool message.
    Args:
    parser: OpenAIResponseParser or AnyscaleResponseParser object
    messages: list of messages in the conversation. The messages with role 'assistant' are ignored
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


def evaluate_gpt(ds, openai_parser):
    num_correct = 0
    pbar = tqdm(total=len(ds), desc="Evaluating GPT4..")
    total_count = 0
    results = []
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
    print("GPT accuracy: ", num_correct / total_count)
    return results

def evaluate_finetuned(ds, anyscale_parser):
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
    print("Anyscale accuracy: ", anyscale_accuracy / total_count)
    return results
