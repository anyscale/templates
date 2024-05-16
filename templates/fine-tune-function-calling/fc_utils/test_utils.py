"""
Test data preprocessing utilities
"""

import re
import ray.data

from fc_utils.function_extraction_utils import extract_jsons, TOOL_CALL_TAGS, TOOL_RESULT_TAGS, FunctionCallNotFoundError, FunctionResponseNotFoundError, get_tool_calls_from_response

def extract_tool_result(content):
    escaped_tool_result_tags = [re.escape(tag) for tag in TOOL_RESULT_TAGS]
    # get the tool result in the pattern "RESULT_TAG1 [tool_result] RESULT_TAG2"
    pattern = r"{}\s*\[([\s\S]*?)\]\s*{}".format(*escaped_tool_result_tags)
    match = re.search(pattern, content)
    if match:
        return match.group(1)
    else:
        raise FunctionResponseNotFoundError("Tool result not found in content")

def test_data_mapper(example):
    example["messages"] = list(example["messages"])
    messages = example["messages"]
    example["openai_messages"] = []
    example["anyscale_messages"] = []
    openai_system_msg = {"role": "system", "content": "You are a helpful assistant."} # default
    example["openai_messages"].append(openai_system_msg)
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
            if TOOL_RESULT_TAGS[0] in message["content"]:
                # openai's message format requires the function name. which is only present in the previous assistant response.
                fn_call_str = previous_message["content"].replace("'", "")
                fn_calls = extract_jsons(fn_call_str)
                if not len(fn_calls):
                    raise FunctionCallNotFoundError("No function call found in the previous assistant response")
                tool_call = fn_calls[0]
                openai_message = {
                        "role": "tool",
                        "name": tool_call["name"],
                        "content": extract_tool_result(messages[i]["content"]),
                    }
                example["openai_messages"].append(openai_message)
            else:
                example["openai_messages"].append(message)
        else:
            content = messages[i]["content"]
            tool_calls = None
            if TOOL_CALL_TAGS[0] in messages[i]["content"]:
                content, tool_calls = get_tool_calls_from_response(messages[i]["content"])
            expected_response = {"role": messages[i]["role"], "content": content, "tool_calls": tool_calls}
            example["expected_responses"].append(expected_response)
            example["openai_messages"].append(message)

    return example



def get_evaluation_dataset(test_ds: ray.data.Dataset):
    # converts test_ds to a list of dicts because the nested structure of the modified dataset is not supported by pyArrow
    modified_ds = []
    for example in test_ds.iter_rows():
        modified_ds.append(test_data_mapper(example))
    return modified_ds
