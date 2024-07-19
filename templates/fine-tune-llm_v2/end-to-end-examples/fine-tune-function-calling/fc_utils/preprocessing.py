"""
Preprocessing utils for Glaive AI's function calling dataset
"""

from typing import Dict, Any, List

import re
import json
import logging
from pathlib import Path

import ray.data

from fc_utils.function_extraction_utils import (
    get_tool_calls_from_response,
    extract_functions_from_system_msg,
    FunctionCallFormatError,
    DatasetFormat,
    FunctionFormatError,
)
from fc_utils.data_format import (
    GLAIVEAI_SYSTEM_NO_TOOLS,
    GLAIVEAI_SYSTEM_WITH_TOOLS,
    GLAIVEAI_TOOL_CALL_INDICATORS,
    GLAIVEAI_TOOL_CALL_PREFIX,
    GLAIVEAI_EOS,
    DEFAULT_SYSTEM_PROMPT,
    GlaiveAIRoleTags,
    MessageType,
    TOOL_CALL_TAGS,
    TOOL_RESULT_TAGS,
    TOOL_LIST_TAGS,
)


class InvalidSystemPromptError(Exception):
    """Raised when an invalid system prompt is found."""

    pass


class InvalidRoleError(Exception):
    """Raised when an invalid role is found in a message."""

    pass


class TagsNotFoundError(Exception):
    """Raised when none of the expected tags are found in the chat string."""

    pass


def _glaive_to_openai(example: Dict[str, Any]) -> Dict[str, Any]:
    """Mapper function to process the glaive ai function calling dataset into the OpenAI format."""
    messages = []
    tools = None
    if GLAIVEAI_SYSTEM_WITH_TOOLS in example["system"]:
        try:
            tools = extract_functions_from_system_msg(
                example["system"], format=DatasetFormat.GLAIVE
            )
        except FunctionFormatError as e:
            logging.info(f"Error processing example {example['system']} : {e}")
            return {"messages": None, "tools": None}
        # Convert to string for compatiblity with PyArrow
        tools = json.dumps(tools)
    elif GLAIVEAI_SYSTEM_NO_TOOLS not in example["system"]:
        # If an unexpected system prompt is found, raise an error to investigate
        raise InvalidSystemPromptError(
            f"System prompt {example['system']} does not match expected prefixes"
        )

    messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    try:
        chat_messages = chat_str_to_messages(example["chat"])
    except (FunctionCallFormatError, TagsNotFoundError, json.JSONDecodeError) as e:
        # For chat data format errors, propagate None types for filtering later
        logging.info(f"Error processing example {example['chat']} : {e}")
        return {"messages": None, "tools": None}

    messages.extend(chat_messages)
    processed_example = {"messages": messages, "tools": tools}
    return processed_example


def combine_multiple_entries(assistant_content: str) -> str:
    """Combines multiple entries of the assistant role into one entry when the function call is split into multiple entries."""
    if (
        assistant_content.startswith(GLAIVEAI_TOOL_CALL_PREFIX)
        or GLAIVEAI_TOOL_CALL_PREFIX not in assistant_content
    ):
        return assistant_content
    else:
        assistant_tag = GlaiveAIRoleTags.ASSISTANT.value
        fn_call_pattern = r"([\s\S]*?){}\s+{}([\s\S]*)".format(
            re.escape(assistant_tag), re.escape(GLAIVEAI_TOOL_CALL_PREFIX)
        )
        function_call_match = re.search(fn_call_pattern, assistant_content, re.DOTALL)
        if function_call_match:
            content1 = function_call_match.group(1).strip()
            content2 = function_call_match.group(2).strip()
            assistant_content = content1 + GLAIVEAI_TOOL_CALL_PREFIX + content2
    return assistant_content


def chat_str_to_messages(chat: str) -> List[MessageType]:
    """Helper function to convert the chat string in the Glaive format into a list of messages in the OpenAI format.

    Args:
        chat: The chat string
        tool_to_user: Boolean indicating if the tool response should be converted to user role

    Returns:
        messages: List of messages in the OpenAI format
    """
    user_tag = GlaiveAIRoleTags.USER.value
    assistant_tag = GlaiveAIRoleTags.ASSISTANT.value
    tool_tag = GlaiveAIRoleTags.TOOL.value
    # Regex pattern to extract user, assistant and tool messages.
    tag_pattern = re.compile(
        rf"(?:{user_tag}\s*(?P<user>.*?)\s*(?={assistant_tag}|$)|{assistant_tag}\s*(?P<assistant>.*?)\s*(?={tool_tag}|{user_tag}|$)|{tool_tag}\s*(?P<function_response>.*?)\s*(?={tool_tag}|{assistant_tag}|$))",
        re.DOTALL,
    )

    matches = tag_pattern.finditer(chat)
    # If no matches found, raise an error
    if not matches:
        raise TagsNotFoundError(f"No user/assistant/tool message found in {chat}")
    messages = []
    # Keep track of the tool call ids and function names in the previous assistant response
    previous_tool_calls_info = []
    # Loop through all matches and extract the respective roles and content
    for match in matches:
        if match.group("user"):
            user_content = match.group("user").strip()
            msg = {"role": "user", "content": user_content}
        elif match.group("assistant"):
            assistant_content = match.group("assistant").strip()
            assistant_content = combine_multiple_entries(assistant_content)

            # Glaive dataset is full of single function calls.
            # We extract the function call and place it in the tool_calls field
            openai_fmt_tool_calls = []
            if GLAIVEAI_TOOL_CALL_PREFIX in assistant_content:
                # Get the function calls from the response.
                # We convert to JSON and then back to string to ensure the format is correct
                assistant_content, tool_calls = get_tool_calls_from_response(
                    assistant_content,
                    GLAIVEAI_TOOL_CALL_INDICATORS,
                    format=DatasetFormat.GLAIVE,
                )
                if assistant_content is None:
                    assistant_content = ""
                for i, tool_call in enumerate(tool_calls):
                    # OpenAI's API assigns a 24-length uuid for a tool call
                    # However, we need not ask the model to generate this
                    # and rather internally track the tool call by index
                    openai_fmt_tool_call = {
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            # arguments field is stringified with single quotes
                            "arguments": json.dumps(tool_call["arguments"]),
                        },
                    }
                    openai_fmt_tool_calls.append(openai_fmt_tool_call)
            # Remove the eos token if present
            assistant_content = assistant_content.replace(GLAIVEAI_EOS, "")
            msg = {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": openai_fmt_tool_calls,
            }
            previous_tool_calls_info = []
            for i, tool_call in enumerate(openai_fmt_tool_calls):
                # To train the model, we can assign a simple incremental id for a tool call
                # This is going to be used in the tool response so that the model
                # can track which tool call it corresponds to
                tool_call_id = f"call_{i+1}"
                previous_tool_calls_info.append(
                    (f"call_{i+1}", tool_call["function"]["name"])
                )
        elif match.group("function_response"):
            function_response = match.group("function_response").strip()
            role = "tool"
            # Get the previous tool call id. Raise an error if no tool call id is found
            if not len(previous_tool_calls_info):
                raise FunctionCallFormatError(
                    "No tool call id found for the tool response"
                )
            tool_call_id, tool_call_name = previous_tool_calls_info.pop(0)
            msg = {
                "role": role,
                "content": function_response,
                "name": tool_call_name,
                "tool_call_id": tool_call_id,
            }
        else:
            # Sometimes, the input can be malformed with no content in the captured group.
            # Example: 'USER: \n'. Skip these entries
            continue
        messages.append(msg)
    return messages


def _openai_to_anyscale(example: Dict[str, Any]) -> Dict[str, Any]:
    """Mapper function to process an example in the OpenAI format into the Anyscale format.

    Formats the tool list in the system message. Formats the tool calls and tool results in plain text with special indicator tags for all.
    """
    assert (
        example["messages"][0]["role"] == "system"
    ), "First message must be from system"
    anyscale_messages = []
    openai_messages = example["messages"]
    tools = example["tools"]
    # Convert the stringified tools to a list of jsons
    tools = json.loads(tools) if tools else []
    for message in openai_messages:
        if message["role"] == "system":
            anyscale_message = {"role": "system", "content": message["content"]}
            if len(tools):
                # Convert to string and add tool list tags
                tools_str = json.dumps(tools)
                tools_str_with_tags = (
                    f"{TOOL_LIST_TAGS.start} {tools_str} {TOOL_LIST_TAGS.end}"
                )
                anyscale_message["content"] += tools_str_with_tags
        elif message["role"] == "assistant":
            tool_calls = message["tool_calls"]
            anyscale_message = {"role": "assistant", "content": message["content"]}
            if isinstance(tool_calls, list) and len(tool_calls):
                # Convert list of tool_calls to string and add tool call tags
                tool_calls_str = json.dumps(tool_calls)
                tool_calls_str_with_tags = (
                    f"{TOOL_CALL_TAGS.start} {tool_calls_str} {TOOL_CALL_TAGS.end}"
                )
                anyscale_message["content"] += tool_calls_str_with_tags
        elif message["role"] == "tool":
            # Convert tool result to string and add tool result tags
            tool_result_str = json.dumps(
                {
                    "name": message["name"],
                    "content": message["content"],
                    "tool_call_id": message["tool_call_id"],
                }
            )
            tool_result_str_with_tags = (
                f"{TOOL_RESULT_TAGS.start} {tool_result_str} {TOOL_RESULT_TAGS.end}"
            )
            # Convert to user role
            anyscale_message = {"role": "user", "content": tool_result_str_with_tags}
        elif message["role"] == "user":
            anyscale_message = {"role": "user", "content": message["content"]}
        else:
            InvalidRoleError(f"Invalid role {message['role']} found in the messages")
        anyscale_messages.append(anyscale_message)
    # If the last message is from the user, drop it
    if anyscale_messages[-1]["role"] == "user":
        anyscale_messages = anyscale_messages[:-1]
    return {"messages": anyscale_messages}


def filter_func(example: Dict[str, Any]) -> bool:
    """Simple filter function that returns False if the message list has two consecutive messages
    from the same role, and True otherwise.

    This is to remove erraneous entries where the function call can be repeated in consecutive entries. Example:
    {.......'content': 'Sure, let me help you with that. <functioncall> {"name": "track_package", .....} ', 'role': 'assistant'}, {'content': '<functioncall> {"name": "track_package", ....} ', 'role': 'assistant'}, .....}
    """
    messages = example["messages"]
    is_good_entry = True
    j = 0
    while j + 1 < len(messages):
        # Sometimes,a single message has the same assistant response repeated. We remove these entries along with the ones where we have consecutive assistant responses
        if (
            messages[j]["role"] == messages[j + 1]["role"]
            or GlaiveAIRoleTags.ASSISTANT.value in messages[j]["content"]
        ):
            is_good_entry = False
            break

        j += 1
    return is_good_entry


def openai_to_anyscale(ray_ds: ray.data.Dataset) -> ray.data.Dataset:
    """Preprocesses the input dataset into an Anyscale compatible format .

    The input dataset is expected to be in the OpenAI messages format.
    """
    ray_ds = ray_ds.map(_openai_to_anyscale)
    return ray_ds


def glaive_to_openai(ray_ds: ray.data.Dataset) -> ray.data.Dataset:
    """Preprocesses the input GlaiveAI dataset into the OpenAI format"""
    ray_ds = ray_ds.map(_glaive_to_openai)
    ray_ds = ray_ds.filter(lambda x: x["messages"] is not None)
    ray_ds = ray_ds.filter(filter_func)
    return ray_ds


def save_to_jsonl(ds: ray.data.Dataset, filepath: str) -> None:
    """Saves a Ray dataset to a jsonl file."""

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df = ds.to_pandas()
    df.to_json(filepath, orient="records", lines=True)
