"""
Preprocessing utils for Glaive AI's function calling dataset
"""

import re
import json
from typing import Dict, Any, List, Optional
import ray.data
from fc_utils.function_extraction_utils import (
    get_tool_calls_from_response,
    extract_functions_from_system_msg,
    FunctionCallNotFoundError,
    DatasetFormat,
)
from fc_utils.data_format import *
import logging
import uuid


class InvalidSystemPromptError(Exception):
    pass


class InvalidRoleError(Exception):
    pass


class TagsNotFoundError(Exception):
    pass


def glaive_to_openai(example: Dict[str, Any]) -> Dict[str, Any]:
    """Mapper function to process the glaive ai function calling dataset into the OpenAI format."""
    messages = []
    tools = None
    if GLAIVEAI_SYSTEM_WITH_TOOLS in example["system"]:
        tools = extract_functions_from_system_msg(
            example["system"], format=DatasetFormat.GLAIVE
        )
        # # convert to string for compatiblity with PyArrow
        tools = json.dumps(tools)
    elif GLAIVEAI_SYSTEM_NO_TOOLS not in example["system"]:
        # If an unexpected system prompt is found, raise an error to investigate
        raise InvalidSystemPromptError(
            f"System prompt {example['system']} does not match expected prefixes"
        )

    messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    try:
        chat_messages = chat_str_to_messages(example["chat"])
    except (FunctionCallNotFoundError, TagsNotFoundError, json.JSONDecodeError) as e:
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
        fn_call_pattern = r"([\s\S]*?)ASSISTANT: {}([\s\S]*)".format(
            re.escape(GLAIVEAI_TOOL_CALL_PREFIX)
        )
        function_call_match = re.search(fn_call_pattern, assistant_content, re.DOTALL)
        if function_call_match:
            content1 = function_call_match.group(1).strip()
            content2 = function_call_match.group(2).strip()
            assistant_content = content1 + GLAIVEAI_TOOL_CALL_PREFIX + content2
    return assistant_content


def chat_str_to_messages(chat: str) -> List[MessageType]:
    """Helper function to convert the chat string into a list of messages with roles.

    Args:
        chat: The chat string
        tool_to_user: Boolean indicating if the tool response should be converted to user role

    Returns:
        messages: List of messages with roles and content
    """
    # Regex pattern to extract user, assistant and tool messages.
    tag_pattern = re.compile(
        r"(?:USER:\s*(?P<user>.*?)\s*(?=ASSISTANT|$)|ASSISTANT:\s*(?P<assistant>.*?)(?=\n\n\nFUNCTION RESPONSE|\n*(?=USER|$))|\n\n\nFUNCTION RESPONSE:\s*(?P<function_response>.*?)\s*(?=ASSISTANT|USER|$))",
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
            # We extract the single function call and place it in the tool_calls field
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
                    # However, to train the model, we can assign a simple incremental id
                    # This is also going to be used in the tool response
                    tool_call_id = f"call_{i+1}"
                    openai_fmt_tool_call = {
                        "id": tool_call_id,
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
            previous_tool_calls_info = [(tool_call["id"], tool_call["function"]["name"]) for tool_call in openai_fmt_tool_calls]
        elif match.group("function_response"):
            function_response = match.group("function_response").strip()
            role = "tool"
            # Get the previous tool call id. Raise an error if no tool call id is found
            if not len(previous_tool_calls_info):
                raise FunctionCallNotFoundError(
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


def openai_to_anyscale(example: Dict[str, Any]) -> Dict[str, Any]:
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
                tools_str = json.dumps(tools)
                tools_str_with_tags = (
                    f"{TOOL_LIST_TAGS[0]} {tools_str} {TOOL_LIST_TAGS[1]}"
                )
                anyscale_message["content"] += tools_str_with_tags
        elif message["role"] == "assistant":
            # Convert list of tool_calls to string and add tool call tags
            tool_calls = message["tool_calls"]
            anyscale_message = {"role": "assistant", "content": message["content"]}
            if isinstance(tool_calls, list) and len(tool_calls):
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
    # if the last message is from the user, drop it
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
        # sometimes,a single message has the same assistant response repeated. We remove these entries along with the ones where we have consecutive assistant responses
        if (
            messages[j]["role"] == messages[j + 1]["role"]
            or GlaiveAIRoleTags.ASSISTANT.value in messages[j]["content"]
        ):
            is_good_entry = False
            break

        j += 1
    return is_good_entry


def preprocess_to_anyscale_format(ray_ds: ray.data.Dataset) -> ray.data.Dataset:
    """Preprocesses the input dataset into an Anyscale Endpoints-compatible format .

    The input dataset is expected to be in the OpenAI messages format.
    """
    ray_ds = ray_ds.map(openai_to_anyscale)
    # Filter for good measure
    ray_ds = ray_ds.filter(filter_func)
    return ray_ds


def preprocess_to_openai_format(ray_ds: ray.data.Dataset) -> ray.data.Dataset:
    """Preprocesses the input GlaiveAI dataset into the OpenAI format"""
    ray_ds = ray_ds.map(glaive_to_openai)
    ray_ds = ray_ds.filter(lambda x: x["messages"] is not None)
    ray_ds = ray_ds.filter(filter_func)
    return ray_ds


def pprint_example(example: Dict[str, Any], dataset_format: DatasetFormat) -> None:
    """Pretty prints an example with colors for different roles."""
    pprint_str = ""
    # ANSI escape code for blue, green, red, yellow and magenta colors
    blue = "\033[94m"
    reset = "\033[0m"
    green = "\033[92m"
    red = "\033[91m"
    magenta = "\033[95m"
    yellow = "\033[93m"
    colors = {
        "system": red,
        "user": green,
        "assistant": blue,
        "tool": yellow,
        "tools": magenta,
        "chat": green,
    }
    for key in example.keys():
        if key == "messages":
            pprint_str += f"{colors['chat']}Messages: {reset}\n"
            for msg in example["messages"]:
                role = msg["role"]
                content = msg["content"]
                color = colors.get(role, reset)
                string = f"\t{color}{role}: {reset}{content}\n"
                if role == "assistant" and dataset_format == DatasetFormat.OPENAI:
                    tool_calls = msg.get("tool_calls", "")
                    string = f"\t{color}{role}: \n\t\tcontent: {reset}{content}\n"
                    string += f"\t\t{color}tool_calls: {reset}{tool_calls}\n"
                elif role == "tool":
                    name = msg.get("name", "")
                    if dataset_format == DatasetFormat.OPENAI:
                        response_str = json.dumps(
                            {
                                "name": name,
                                "content": content,
                                "tool_call_id": msg["tool_call_id"],
                            }
                        )
                    else:
                        response_str = content
                    string = f"\t{color}{role}: {reset}{response_str}\n"
                pprint_str += string
        else:
            color = colors.get(key, reset)
            string = f"{color}{key.capitalize()}: {reset}{example[key]}\n"
            pprint_str += string
    print(pprint_str)


def save_to_jsonl(ds: ray.data.Dataset, filepath: str) -> None:
    """Saves a Ray dataset to a jsonl file."""
    df = ds.to_pandas()
    df.to_json(filepath, orient="records", lines=True)
