"""
Test data preprocessing utilities
"""

import re
import ray.data
from typing import Tuple, Dict, Any

from fc_utils.function_extraction_utils import extract_jsons, FunctionCallNotFoundError, FunctionResponseNotFoundError, get_tool_calls_from_response

def extract_tool_result(content: str, tool_result_tags: Tuple[str, str]) -> str:
    """
    Extracts the tool result from the content.

    Args:
        content: The content to extract the tool result from
        tool_result_tags: Tuple containing the start and end tags for the tool result

    Returns:
        tool_result: The extracted tool result
    """
    escaped_tool_result_tags = [re.escape(tag) for tag in tool_result_tags]
    # get the tool result in the pattern "RESULT_TAG1 [tool_result] RESULT_TAG2"
    pattern = r"{}\s*\[([\s\S]*?)\]\s*{}".format(*escaped_tool_result_tags)
    match = re.search(pattern, content)
    if match:
        return match.group(1)
    else:
        raise FunctionResponseNotFoundError("Tool result not found in content")

def test_data_mapper(example: Dict[str, Any], tool_call_tags: Tuple[str, str], tool_result_tags: Tuple[str, str]) -> Dict[str, Any]:
    """
    Preprocesses a test data sample for evaluation.

    The test example is processed to have the following fields:
    1. `openai_messages` : the list of messages in the conversation for GPT-4. The user messages will be fed to the model one by one. The function responses are designated with the role "tool".
    2. `anyscale_messages`: the list of messages in the conversation for our fine-tuned model.
    3. `tools`: the list of tools to pass to the OpenAI model.
    4. `expected_responses` : the list of ground truth assistant responses in the conversation.

    Args:
        example: The test example to preprocess
        tool_call_tags: Tuple containing the start and end tags for the tool call
        tool_result_tags: Tuple containing the start and end tags for the tool result

    Returns:
        example: The preprocessed test example
    """

    example["messages"] = list(example["messages"])
    messages = example["messages"]
    example["openai_messages"] = []
    example["anyscale_messages"] = []
    # default system message
    openai_system_msg = {"role": "system", "content": "You are a helpful assistant."}
    example["openai_messages"].append(openai_system_msg)
    # messages to the anyscale model are already in the expected format
    example["anyscale_messages"] = messages
    functions = extract_jsons(messages[0]["content"])
    # Some functions may not have parameters. Fix them
    for fn in functions:
        if not fn["parameters"]:
            fn["parameters"] = {
                "type": "object",
                "properties": {},
                "required": [],
            }
    example["tools"] = functions
    example["expected_responses"] = []

    for i in range(1, len(messages)):
        message = messages[i]
        previous_message = messages[i-1]
        if message["role"] == "user":
            if tool_result_tags[0] in message["content"]:
                # openai's message format requires the function name. which is only present in the previous assistant response.
                fn_call_str = previous_message["content"].replace("'", "")
                fn_calls = extract_jsons(fn_call_str)
                if not len(fn_calls):
                    raise FunctionCallNotFoundError("No function call found in the previous assistant response")
                tool_call = fn_calls[0]
                openai_message = {
                        "role": "tool",
                        "name": tool_call["name"],
                        "content": extract_tool_result(messages[i]["content"], tool_result_tags),
                    }
                example["openai_messages"].append(openai_message)
            else:
                example["openai_messages"].append(message)
        else:
            content = messages[i]["content"]
            tool_calls = None
            if tool_call_tags[0] in messages[i]["content"]:
                content, tool_calls = get_tool_calls_from_response(messages[i]["content"], tool_call_tags=tool_call_tags)
            expected_response = {"role": messages[i]["role"], "content": content, "tool_calls": tool_calls}
            example["expected_responses"].append(expected_response)
            example["openai_messages"].append(message)

    return example



def get_evaluation_dataset(test_ds: ray.data.Dataset, tool_call_tags: Tuple[str, str], tool_result_tags: Tuple[str, str]):
    """
    Handles the preprocessing of the test dataset for evaluation.

    Args:
        test_ds: The test dataset to preprocess
        tool_call_tags: Tuple containing the start and end tags for the tool call
        tool_result_tags: Tuple containing the start and end tags for the tool result

    Returns:
        modified_ds: The preprocessed test dataset
    """
    # converts test_ds to a list of dicts because the nested structure of the modified dataset is not supported by PyArrow
    modified_ds = []
    for example in test_ds.iter_rows():
        modified_ds.append(test_data_mapper(example, tool_call_tags, tool_result_tags))
    return modified_ds
