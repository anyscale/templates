"""
Function call json extraction utils.
"""

import json
import re
from typing import List, Dict, Any, Tuple, TypeAlias, Union


# define our custom type for a tool call
ToolCallType: TypeAlias = Dict[str, Union[str, Dict[str, str]]]

class FunctionCallNotFoundError(Exception):
    pass

class FunctionResponseNotFoundError(Exception):
    pass

def extract_jsons(string: str) -> List[Dict[str, Any]]:
    """
    Extracts JSON objects from a string containing one or more JSONs.
    Example: "[{"name": "weather", "arguments": {\"location\": \"New York\"}}]"
    """
    json_strs = []
    brace_count = 0
    start = None
    # flag to check if we are inside a string
    in_string = False
    # flag to handle an escape sequence
    escape = False
    for i, char in enumerate(string):
        if char == '"' and not escape:
            # Toggle the in_string flag on encountering a quote
            in_string = not in_string
        if in_string:
            # Handle escape sequences within strings
            escape = char == '\\' and not escape
            # Skip further processing if inside a string
            continue

        if char == '{':
            if brace_count == 0:
                start = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                json_strs.append(string[start:end])

    jsons = [json.loads(json_str) for json_str in json_strs]

    return jsons

def get_tool_calls_from_response(assistant_content: str, tool_call_tags: Tuple[str, str]) -> Tuple[str, List[ToolCallType]]:
    """
    Extracts tool calls from the assistant response.

    Args:
        assistant_content: The assistant response content
        tool_call_tags: Tuple containing the start and end tags for the tool call

    Returns:
        assistant_content: The assistant response content
        tool_calls: List of tool calls extracted from the assistant response
    """
    # remove trailing whitespaces
    assistant_content = assistant_content.strip()
    escaped_tool_call_tags = [re.escape(tag) for tag in tool_call_tags]
    if assistant_content.startswith(tool_call_tags[0]):
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
        # Replace single quotes with triple double quotes
        function_call_str = function_call_str.replace(
            "'", ""
        )

        tool_calls = extract_jsons(function_call_str)
    else:
        raise FunctionCallNotFoundError("No function call found in assistant response")

    return assistant_content, tool_calls
