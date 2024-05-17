"""
Preprocessing utils for Glaive AI's function calling dataset
"""

import re
import json
from typing import Dict, Any, List
import ray.data

tags = {"user": "USER: ", "assistant": "ASSISTANT: ", "tool": "FUNCTION RESPONSE: "}

TOOL_RESULT_TAGS = ["[TOOL_RESULTS]", "[/TOOL_RESULTS]"]
TOOL_CALL_TAGS = ["[TOOL_CALLS]", "[/TOOL_CALLS]"]
TOOL_LIST_TAGS = ["[TOOL_LIST]", "[/TOOL_LIST]"]

GLAIVEAI_TEMPLATE = {
    "system_prefixes": [
        "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -",
        "SYSTEM: You are a helpful assistant, with no access to external functions.",
    ],
    "tool_call_prefix": "<functioncall>",
    "eos": "<|endoftext|>",
}


class InvalidSystemPromptError(Exception):
    pass


class TagsNotFoundError(Exception):
    pass

def extract_functions_from_system_msg(system_str: str) -> List[Dict[str, Any]]:
    """
    Extracts the functions from the system message with a simple regex pattern. If
    the function is not a valid JSON, it is skipped.
    Args:
        system_str: The system message
    Returns:
        functions: List of functions successfully extracted from the system message
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
    return functions

def initial_mapper(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mapper function to process the glaive ai function calling dataset into the OpenAI format
    """
    messages = []
    system_str = "You are a helpful assistant."
    tools = None
    system_prompt_prefixes = GLAIVEAI_TEMPLATE["system_prefixes"]
    if system_prompt_prefixes[0] in example["system"]:
        tools = extract_functions_from_system_msg(example["system"])
        # convert to string
        tools = json.dumps(tools)
    elif system_prompt_prefixes[1] not in example["system"]:
        raise InvalidSystemPromptError(
            f"System prompt {example['system']} does not match expected prefixes"
        )

    messages.append({"role": "system", "content": system_str})
    chat_messages = chat_str_to_messages(example["chat"])
    messages.extend(chat_messages)
    example["messages"] = messages
    example["tools"] = tools
    return example


def combine_multiple_entries(assistant_content: str) -> str:
    """
    Combines multiple entries of the assistant role into one entry when the function call is split into multiple entries.
    """
    if assistant_content.startswith(GLAIVEAI_TEMPLATE["tool_call_prefix"]) or GLAIVEAI_TEMPLATE["tool_call_prefix"] not in assistant_content:
        return assistant_content
    else:
        fn_call_pattern = r"([\s\S]*?)ASSISTANT: {}([\s\S]*)".format(re.escape(GLAIVEAI_TEMPLATE["tool_call_prefix"]))
        function_call_match = re.search(fn_call_pattern, assistant_content, re.DOTALL)
        if function_call_match:
            content1 = function_call_match.group(1).strip()
            content2 = function_call_match.group(2).strip()
            assistant_content = content1 + GLAIVEAI_TEMPLATE["tool_call_prefix"] + content2
    return assistant_content



def chat_str_to_messages(chat: str, tool_to_user=False) -> List[Dict[str, str]]:
    """
    Helper function to convert the chat string into a list of messages with roles.
    Args:
        chat: The chat string
        tool_to_user: Boolean indicating if the tool response should be converted to user role
    Returns:
        messages: List of messages with roles and content
    """
    # regex pattern to extract user, assistant and tool messages.
    tag_pattern = re.compile(
        r"(?:USER:\s*(?P<user>.*?)\s*(?=ASSISTANT|$)|ASSISTANT:\s*(?P<assistant>.*?)(?=\n\n\nFUNCTION RESPONSE|\n*(?=USER|$))|\n\n\nFUNCTION RESPONSE:\s*(?P<function_response>.*?)\s*(?=ASSISTANT|USER|$))",
        re.DOTALL,
    )

    matches = tag_pattern.finditer(chat)
    user_content = assistant_content = None
    # if no matches found, raise an error
    if not matches:
        raise TagsNotFoundError(f"No user/assistant/tool message found in {chat}")

    messages = []
    # Loop through all matches and extract the respective roles and content
    for match in matches:
        if match.group("user"):
            user_content = match.group("user").strip()
            msg = {"role": "user", "content": user_content}
        elif match.group("assistant"):
            assistant_content = match.group("assistant").strip()
            assistant_content = combine_multiple_entries(assistant_content)
            # glaive dataset is full of single function calls.
            # We convert the single function call into a list and add tool call tags
            if GLAIVEAI_TEMPLATE["tool_call_prefix"] in assistant_content:
                # make function call a list and add tags
                assistant_content = assistant_content.replace(
                    GLAIVEAI_TEMPLATE["tool_call_prefix"], f"{TOOL_CALL_TAGS[0]} ["
                )
                assistant_content += f"] {TOOL_CALL_TAGS[1]}"
            assistant_content = assistant_content.replace(GLAIVEAI_TEMPLATE["eos"], "")
            msg = {"role": "assistant", "content": assistant_content}
        elif match.group("function_response"):
            function_response = match.group("function_response").strip()
            role = "tool"
            # convert function response to a list for generality and add tags
            function_response = f"[{function_response}]"
            if tool_to_user:
                # add tool tags only if the tool response is to be converted to user role
                function_response = f"{TOOL_RESULT_TAGS[0]} {function_response} {TOOL_RESULT_TAGS[1]}"
                role = "user"
            msg = {"role": role, "content": function_response}
        messages.append(msg)
    return messages


def final_mapper(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mapper function to process the glaive ai function calling dataset into Anyscale Endpoints compatible format.
    Preprocessing steps:
    1. Remove the used special tokens "USER: "<|endoftext|>" and bring them to a general format (since different models will have different special tokens for roles, end of text, etc).
    2. Process tool responses into "user" role. "FUNCTION RESPONSE: " entries are processed into the role "user"
    """
    messages = []
    system_str = "You are a helpful assistant."
    tools = ""
    system_prompt_prefixes = GLAIVEAI_TEMPLATE["system_prefixes"]
    if system_prompt_prefixes[0] in example["system"]:
        tools = extract_functions_from_system_msg(example["system"])
         # convert to string and add tags
        tools_str= json.dumps(tools)
        tools = f"{TOOL_LIST_TAGS[0]} {tools_str} {TOOL_LIST_TAGS[1]}"
    elif system_prompt_prefixes[1] not in example["system"]:
        raise InvalidSystemPromptError(
            f"System prompt {example['system']} does not match expected prefixes"
        )

    messages.append({"role": "system", "content": system_str + tools})
    chat_messages = chat_str_to_messages(example["chat"], tool_to_user=True)
    messages.extend(chat_messages)
    if messages[-1]["role"] != "assistant":
        messages = messages[:-1]  # drop last message if from user

    example["messages"] = messages
    return example


def filter_func(example: Dict[str, Any]) -> bool:
    """
    Simple filter function that returns False if the message list has two consecutive messages
    from the same role. If otherwise, the function returns True.
    This is to remove erraneous entries that can look like:
    {.......'content': 'Sure, let me help you with that. <functioncall> {"name": "track_package", "arguments": \'{"tracking_number": "123456789"}\'} ', 'role': 'assistant'}, {'content': '<functioncall> {"name": "track_package", "arguments": \'{"tracking_number": "123456789"}\'} ', 'role': 'assistant'.....}
    """
    messages = example["messages"]
    is_good_entry = 1
    j = 0
    while j + 1 < len(messages):
        # sometimes,a single message has the same assistant response repeated. We remove these entries along with the ones where we have consecutive assistant responses
        if messages[j]["role"] == messages[j + 1]["role"] or "ASSISTANT: " in messages[j]["content"]:
            is_good_entry = 0
            break

        j += 1
    return is_good_entry


def preprocess(ray_ds: ray.data.Dataset) -> ray.data.Dataset:
    """
    Preprocesses the Ray dataset into the messages format required by Anyscale Endpoints.
    """
    ray_ds = ray_ds.map(final_mapper)
    ray_ds = ray_ds.filter(filter_func)
    ray_ds = ray_ds.drop_columns(["system", "chat"])  # drop the original columns
    return ray_ds


def pprint_example(example: Dict[str, Any],keys: List[str] = []) -> None:
    """
    Pretty prints an example with colors for different roles
    """
    if not keys:
        keys = example.keys()
    pprint_str = ""
    # ANSI escape code for blue, green, red, yellow and magenta colors
    blue = "\033[94m"
    reset = "\033[0m"
    green = "\033[92m"
    red = "\033[91m"
    magenta = "\033[95m"
    yellow = "\033[93m"
    for key in keys:
        if key == "messages":
            for msg in example["messages"]:
                common_str = f'{msg["role"]}: {reset}{msg["content"]}\n'
                if msg["role"] == "system":
                    # system word is colored blue
                    pprint_str += f"{red}" + common_str
                elif msg["role"] == "user":
                    pprint_str += f"{green}" + common_str
                elif msg["role"] == "assistant":
                    pprint_str += f"{blue}" + common_str
                else:
                    pprint_str += f"{yellow}" + common_str
        elif key == "tools":
            pprint_str += f'{magenta}Tool list: {reset}{example["tools"]}\n'
        elif key == "system":
            pprint_str += f"{blue}{key}: {reset}{example[key]}\n"
        else:
            pprint_str += f"{green}{key}: {reset}{example[key]}\n"
    print(pprint_str)


def save_to_jsonl(ds: ray.data.Dataset, filepath: str) -> None:
    """
    Saves a Ray dataset to a jsonl file
    """
    df = ds.to_pandas()
    df.to_json(filepath, orient="records", lines=True)
