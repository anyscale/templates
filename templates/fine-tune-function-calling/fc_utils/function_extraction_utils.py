"""
Function call json extraction utils.
"""

import json
import re
from typing import List, Dict, Any, Tuple, Union
from fc_utils.data_format import IndicatorTags, ToolCallType, DatasetFormat
from enum import Enum


class FunctionCallNotFoundError(Exception):
    pass


class FunctionResponseNotFoundError(Exception):
    pass


def parse_function_calls(string: str, format: DatasetFormat) -> List[Dict[str, Any]]:
    """Parses the function calls from the string based on the dataset format provided.

    Args:
        string: The string containing the function calls.
        format: The format used to stringify the function calls. One of FunctionCallFormat.GLAIVE or FunctionCallFormat.OPENAI.

    Returns:
        json_list: List of JSONs representing the function calls.
    """
    if format == DatasetFormat.GLAIVE:
        return parse_function_calls_glaive(string)
    else:
        return parse_function_calls_openai(string)


def parse_function_calls_glaive(string: str) -> List[Dict[str, Any]]:
    """
    Parses assistant function calls in the GlaiveAI dataset format into a list of JSONs

    Each json string is expected to have the all fields in double quotes, with the "arguments" field as a stringified JSON object in single quotes.
    Example: "[{\"name\": \"weather\", \"arguments\": '{\"location\": \"New York\"}'}]"

    Args:
        string: The string containing the function calls.

    Returns:
        json_list: List of JSONs representing the function calls.
    """
    # Remove single quotes used for the arguments field.
    string = string.replace("'", "")
    # Parse the string into a list of JSONs
    json_list = json.loads(string)
    if isinstance(json_list, dict):
        json_list = [json_list]
    return json_list


def parse_function_calls_openai(string: str) -> List[Dict[str, Any]]:
    """
    Parses assistant function calls in the OpenAI/Anyscale format into a list of JSONs.

    Here, each JSON object is expected to have the "arguments" field as a stringified JSON object.
    Example: {"name": "create_reminder", "arguments": "{\"reminder_text\": \"Doctors appointment\"}"}

    Args:
        string: The string containing the function calls.

    Returns:
        json_list: List of JSONs representing the function calls.
    """
    # Parse the string into a list of JSONs
    json_list = json.loads(string)
    if isinstance(json_list, dict):
        json_list = [json_list]
    for json_obj in json_list:
        json_obj["arguments"] = json.loads(json_obj["arguments"])
    return json_list


def get_tool_calls_from_response(
    raw_response: str, tool_call_tags: IndicatorTags, format: DatasetFormat
) -> Tuple[str, List[ToolCallType]]:
    """
    Extracts tool calls from the assistant response, and returns the regular text content and the (parsed) tool calls.

    If no tool calls are found, a FunctionCallNotFoundError is raised.

    Args:
        raw_response: The raw response from the assistant
        tool_call_tags: Tuple containing the start and end tags for the tool call
        format: The format used to stringify the function calls. One of FunctionCallFormat.GLAIVE or FunctionCallFormat.OPENAI.

    Returns:
        response_text: The regular text content from the response
        tool_calls: List of tool calls extracted from the assistant response.
    """
    # remove trailing whitespaces
    raw_response = raw_response.strip()
    escaped_tool_call_tags = [re.escape(tag) for tag in tool_call_tags]
    if raw_response.startswith(tool_call_tags.start):
        fn_call_pattern = r"{}([\s\S]*){}".format(*escaped_tool_call_tags)
        extract_content = False
    else:
        fn_call_pattern = r"([\s\S]*?){}([\s\S]*){}".format(*escaped_tool_call_tags)
        extract_content = True
    # Extract the function call information
    function_call_match = re.search(fn_call_pattern, raw_response)
    # Correcting the JSON string format
    if function_call_match:
        if extract_content:
            response_text = function_call_match.group(1).strip()
            function_call_str = function_call_match.group(2).strip()
        else:
            response_text = None
            function_call_str = function_call_match.group(1).strip()

        tool_calls = parse_function_calls(function_call_str, format)
    else:
        raise FunctionCallNotFoundError("No function call found in assistant response")

    return response_text, tool_calls


if __name__ == "__main__":
    ex1 = '[{"name": "weather", "arguments": \'{"location": "New York"}\'}]'
    ex2 = '[{"name": "weather", "arguments": \'{"location": "New York"}\', "name": "get_current_time", "arguments": \'{"location": "New York"}\'}]'
    breakpoint()
    print(parse_function_calls(ex1))
    print(parse_function_calls(ex2))
    print("Done")
