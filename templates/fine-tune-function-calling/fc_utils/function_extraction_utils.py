"""
Function call json extraction utils.
"""

import json
import re
from typing import List, Dict, Any, Tuple, Union, Optional
from fc_utils.data_format import IndicatorTags, ToolCallType, DatasetFormat
from enum import Enum


class FunctionCallNotFoundError(Exception):
    pass


class FunctionResponseNotFoundError(Exception):
    pass


def extract_functions_from_system_msg(
    system_str: str,
    format: DatasetFormat,
    tool_list_tags: Optional[IndicatorTags] = None,
) -> List[Dict[str, Any]]:
    """Extracts functions from the system message based on the dataset format."""
    if format == DatasetFormat.GLAIVE:
        return _extract_functions_from_system_msg_glaive(system_str)
    elif format == DatasetFormat.ANYSCALE:
        return _extract_functions_from_system_msg_anyscale(system_str, tool_list_tags)
    else:
        raise NotImplementedError(
            f"Function extraction for format {format} not implemented"
        )


def _extract_functions_from_system_msg_glaive(system_str: str) -> List[Dict[str, Any]]:
    """Extracts the functions from the system message with a simple regex pattern.

    If the function is not a valid JSON, it is skipped.

    Args:
        system_str: The system message

    Returns:
        functions: List of functions successfully extracted from the system message in the OpenAI format
    """
    # Extracting the functions using regex
    functions_match = re.findall(r"\{.*?\}(?=\s*\{|\s*$)", system_str, re.DOTALL)
    functions = []

    for fn in functions_match:
        try:
            # Convert string representation of dictionary to actual dictionary
            fn_dict = json.loads(fn)
            functions.append(fn_dict)
        except json.JSONDecodeError:
            # In case the string is not a valid JSON, continue without adding it to the list
            continue

    # Some functions may not have parameters. Fix them
    for fn in functions:
        if not fn["parameters"]:
            fn["parameters"] = {
                "type": "object",
                "properties": {},
                "required": [],
            }
    functions = [{"type": "function", "function": fn} for fn in functions]
    return functions


def _extract_functions_from_system_msg_anyscale(
    system_msg: str, tool_list_tags: IndicatorTags
) -> List[str]:
    """
    Extracts the tool list from the system message in the Anyscale format.

    Args:
        system_msg: The system message to extract the tools from

    Returns:
        tools: The list of tools extracted from the system message
    """
    escaped_tool_list_tags = [re.escape(tag) for tag in tool_list_tags]
    system_msg_format = r"([\s\S]*?){}([\s\S]*){}".format(*escaped_tool_list_tags)
    tool_list_match = re.search(system_msg_format, system_msg)
    tools = []
    if tool_list_match:
        tool_list_str = tool_list_match.group(2).strip()
        tools = json.loads(tool_list_str)
    return tools


def parse_function_calls(string: str, format: DatasetFormat) -> List[Dict[str, Any]]:
    """Parses the function calls from the string based on the dataset format provided.

    Args:
        string: The string containing the function calls.
        format: The format used to stringify the function calls.

    Returns:
        json_list: List of JSONs representing the function calls.
    """
    if format == DatasetFormat.GLAIVE:
        return _parse_function_calls_glaive(string)
    else:
        return _parse_function_calls_openai(string)


def _parse_function_calls_glaive(string: str) -> List[Dict[str, Any]]:
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


def _parse_function_calls_openai(string: str) -> List[Dict[str, Any]]:
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
        json_obj["function"]["arguments"] = json.loads(json_obj["function"]["arguments"])
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
        format: The format used to stringify the function calls.

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
    response_text, tool_calls = None, []
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


def parse_tool_result(string: str, tool_result_tags: Tuple[str, str]) -> str:
    """
    Extracts the tool result from the given string.

    Args:
        string: The input string to extract the tool result from
        tool_result_tags: Tuple containing the start and end tags for the tool result

    Returns:
        tool_result: The extracted tool result
    """
    escaped_tool_result_tags = [re.escape(tag) for tag in tool_result_tags]
    # get the tool result in the pattern "RESULT_TAG1 tool_result RESULT_TAG2"
    pattern = r"{}\s*([\s\S]*?)\s*{}".format(*escaped_tool_result_tags)
    match = re.search(pattern, string)
    if match:
        try:
            result = json.loads(match.group(1))
            return result
        except json.JSONDecodeError:
            raise FunctionResponseNotFoundError("Invalid tool result format")
    else:
        raise FunctionResponseNotFoundError("Tool result not found in content")
