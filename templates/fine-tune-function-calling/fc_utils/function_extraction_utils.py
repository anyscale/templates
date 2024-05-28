"""
Function call json extraction utils.
"""

import json
import re
from typing import List, Dict, Any, Tuple, Union, Optional
from enum import Enum

from fc_utils.data_format import (
    IndicatorTags,
    ToolCallType,
    DatasetFormat,
    check_tool_calls_format,
)


class FunctionCallFormatError(Exception):
    """Raised when a function call is expected but not found/ in a wrong format in the assistant response."""

    pass


class FunctionResponseFormatError(Exception):
    """Raised when a function response is not found/ in a wrong format in the given content."""

    pass


class PatternNotFoundError(Exception):
    """Raised when no content is not found based on the given string and tags."""

    pass


class FunctionFormatError(Exception):
    """Raised when function definition is in the wrong format in the given string."""

    pass


def extract_segment_between_tags(
    string: str, indicator_tags: IndicatorTags
) -> Tuple[Optional[str], str]:
    """
    Parses string in the format <prefix> [indicator_tags.start] <special_content> [indicator_tags.end] or [indicator_tags.start] <special_content> [indicator_tags.end]

    Returns the prefix if present and the content between the tags

    Args:
        string: The input string
        indicator_tags: The tags used to format the string

    Returns:
        prefix: The prefix before the special content, if any
        special_content: The content between the tags
    """
    string = string.strip()
    escaped_tags = [re.escape(tag) for tag in indicator_tags]

    if string.startswith(indicator_tags.start):
        pattern = r"{}([\s\S]*?){}".format(*escaped_tags)
        extract_prefix = False
    else:
        pattern = r"([\s\S]*?){}([\s\S]*?){}".format(*escaped_tags)
        extract_prefix = True

    pattern_match = re.search(pattern, string)
    if not pattern_match:
        raise PatternNotFoundError(
            f"No content found in the string {string} with the given tags {indicator_tags}"
        )
    prefix, special_content = None, None
    if extract_prefix:
        prefix = pattern_match.group(1).strip()
        special_content = pattern_match.group(2).strip()
    else:
        prefix = None
        special_content = pattern_match.group(1).strip()
    return prefix, special_content


def _extract_functions_from_system_msg_glaive(system_str: str) -> List[Dict[str, Any]]:
    """Extracts the functions from the system message with a simple regex pattern.

    If the function is not a valid JSON, an error is raised

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
            # In case the string is not a valid JSON, raise an error
            raise FunctionFormatError(
                f"Tool list not in the correct format in : {system_str}"
            )

    # Some functions may not have parameters. Fix them
    for fn in functions:
        if not fn["parameters"]:
            fn["parameters"] = {
                "type": "object",
                "properties": {},
                "required": [],
            }
    # Bring it into the OpenAI tools format
    functions = [{"type": "function", "function": fn} for fn in functions]
    return functions


def _extract_functions_from_system_msg_anyscale(
    system_msg: str, tool_list_tags: IndicatorTags
) -> List[Dict[str, Any]]:
    """
    Extracts the tool list from the system message in the Anyscale format.

    Args:
        system_msg: The system message to extract the tools from

    Returns:
        tools: The list of tools extracted from the system message
    """
    _, tool_list_str = extract_segment_between_tags(system_msg, tool_list_tags)
    tools = json.loads(tool_list_str)
    if not isinstance(tools, list):
        tools = [tools]
    return tools


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
        if "function" not in json_obj or "arguments" not in json_obj["function"]:
            raise FunctionCallFormatError(
                f"Function call not in the correct format in : {string}"
            )
        json_obj["function"]["arguments"] = json.loads(
            json_obj["function"]["arguments"]
        )
    return json_list


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
    try:
        response_text, tool_calls_str = extract_segment_between_tags(
            raw_response, tool_call_tags
        )
        tool_calls = parse_function_calls(tool_calls_str, format)
    except (PatternNotFoundError, json.JSONDecodeError) as e:
        # Propagate a custom exception for use later
        raise FunctionCallFormatError(f"Tool calls could not be found : {e}")

    if not check_tool_calls_format(tool_calls, format):
        raise FunctionCallFormatError("Tool call is not in the correct format")
    return response_text, tool_calls


def parse_tool_result(string: str, tool_result_tags: Tuple[str, str]) -> Dict[str, Any]:
    """
    Extracts the tool result from the given string.

    Args:
        string: The input string to extract the tool result from
        tool_result_tags: Tuple containing the start and end tags for the tool result

    Returns:
        tool_result: The extracted tool result
    """
    try:
        _, tool_result_str = extract_segment_between_tags(string, tool_result_tags)
        result = json.loads(tool_result_str)
    except (PatternNotFoundError, json.JSONDecodeError) as e:
        # Propagate a custom exception for use later
        raise FunctionResponseFormatError(f"Tool result could not be found : {e}")
    return result
