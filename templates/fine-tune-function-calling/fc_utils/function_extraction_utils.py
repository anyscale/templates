"""
Function call json extraction utils.
Reference: https://gist.github.com/kouroshHakha/6cfbe2bf4aaafc5db733d408044f9902#file-create_test_dataset-py
"""

import json
import re
from fc_utils.preprocessing import TOOL_CALL_TAGS, TOOL_RESULT_TAGS, extract_functions

class FunctionCallNotFoundError(Exception):
    pass

class FunctionResponseNotFoundError(Exception):
    pass

def extract_jsons(jsonl_string):
    """
    Extracts JSON objects from a string containing one or more JSONs.
    Example: [{"name": "weather", "arguments": {\"location\": \"New York\"}}]
    """
    json_strs = []
    brace_count = 0
    start = None

    for i, char in enumerate(jsonl_string):
        if char == '{':
            if brace_count == 0:
                start = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                json_strs.append(jsonl_string[start:end])
    jsons = [json.loads(json_str) for json_str in json_strs]
    return jsons

def get_tool_calls_from_response(assistant_content):
    assistant_content = assistant_content.strip()  # remove trailing whitespaces
    escaped_tool_call_tags = [re.escape(tag) for tag in TOOL_CALL_TAGS]
    if assistant_content.startswith(TOOL_CALL_TAGS[0]):
        fn_call_pattern = r"{}([\s\S]*){}".format(*escaped_tool_call_tags)
        extract_content = False
    else:
        fn_call_pattern = r"([\s\S]*?){}([\s\S]*){}".format(*escaped_tool_call_tags)
        extract_content = True
    # Extract the function call information
    function_call_match = re.search(fn_call_pattern, assistant_content)
    # Correcting the JSON string format
    if function_call_match:
        if extract_content:
            assistant_content = function_call_match.group(1).strip()
            function_call_str = function_call_match.group(2).strip()
        else:
            assistant_content = None
            function_call_str = function_call_match.group(1).strip()
        function_call_str = function_call_str.replace(
            "'", ""
        )  # Replace single quotes with triple double quotes

        tool_calls = extract_jsons(function_call_str)
    else:
        raise FunctionCallNotFoundError("No function call found in assistant response")

    return assistant_content, tool_calls


if __name__ == "__main__":
    assistant_content = " Sure let me get the weather for you [TOOL_CALLS] [{\"name\": \"weather\", \"arguments\": '{\"location\": \"New York\"}'}] [/TOOL_CALLS] "
    assistant_content, tool_calls = get_tool_calls_from_response(assistant_content)
    print(assistant_content, tool_calls)
    assistant_content = "[TOOL_CALLS] [ {\"name\": \"calculate_tip\", \"arguments\": '{\"bill_total\": 50, \"tip_percentage\": 15}'} ] [/TOOL_CALLS]"
    assistant_content, tool_calls = get_tool_calls_from_response(assistant_content)
    print(assistant_content, tool_calls)
    with open("/Users/sumanthrh/Documents/templates/templates/fine-tune-function-calling/glaiveai-function-calling-v2-test.jsonl", "r") as f:
        ds = f.readlines()
    messages  = json.loads(ds[0])["messages"]
    breakpoint()
    for msg in messages:
        if  TOOL_CALL_TAGS[0] in msg["content"]:
            print(msg)
            assistant_content, tool_calls = get_tool_calls_from_response(msg["content"])
            print(assistant_content, tool_calls)
            breakpoint()
