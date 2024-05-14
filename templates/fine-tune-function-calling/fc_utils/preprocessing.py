import re

from fc_utils.function_extraction_utils import extract_functions

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


def initial_mapper(example):
    """
    Mapper function to process the glaive ai function calling dataset into the OpenAI format
    """
    messages = []
    system_str = "You are a helpful assistant."
    tools = None
    system_prompt_prefixes = GLAIVEAI_TEMPLATE["system_prefixes"]
    if system_prompt_prefixes[0] in example["system"]:
        tools = extract_functions(example["system"])
        tools = str(tools)  # convert to string
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


def chat_str_to_messages(chat, tool_to_user=False):
    tag_pattern = re.compile(
        r"(?:USER:\s*(?P<user>.*?)\s*(?=ASSISTANT|$)|ASSISTANT:\s*(?P<assistant>.*?)(?=\n\n\nFUNCTION RESPONSE|\n*(?=USER|$))|\n\n\nFUNCTION RESPONSE:\s*(?P<function_response>.*?)\s*(?=ASSISTANT|USER|$))",
        re.DOTALL,
    )

    chat_str = chat.replace(
        GLAIVEAI_TEMPLATE["eos"], ""
    )  # remove model specific end of text token
    matches = tag_pattern.finditer(chat_str)
    user_content = assistant_content = None
    if not matches:
        raise TagsNotFoundError(f"No user/assistant/tool message found in {chat}")
    messages = []
    for match in matches:
        if match.group("user"):
            user_content = match.group("user").strip()
            msg = {"role": "user", "content": user_content}
        elif match.group("assistant"):
            assistant_content = match.group("assistant").strip()
            if GLAIVEAI_TEMPLATE["tool_call_prefix"] in assistant_content:
                # make function call a list and add tags
                assistant_content = assistant_content.replace(
                    GLAIVEAI_TEMPLATE["tool_call_prefix"], TOOL_CALL_TAGS[0] + " " + "["
                )
                assistant_content += "]" + " " + TOOL_CALL_TAGS[1]
            msg = {"role": "assistant", "content": assistant_content}
        elif match.group("function_response"):
            function_response = match.group("function_response").strip()
            role = "tool"
            function_response = "[" + function_response + "]"
            if tool_to_user:
                function_response = (
                    TOOL_RESULT_TAGS[0]
                    + " "
                    + function_response
                    + " "
                    + TOOL_RESULT_TAGS[1]
                )
                role = "user"
            msg = {"role": role, "content": function_response}
        messages.append(msg)
    return messages


def final_mapper(example):
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
        tools = extract_functions(example["system"])
        tools = (
            TOOL_LIST_TAGS[0] + " " + str(tools) + " " + TOOL_LIST_TAGS[1]
        )  # convert to string and add tags
    elif system_prompt_prefixes[1] not in example["system"]:
        raise InvalidSystemPromptError(
            f"System prompt {example['system']} does not match expected prefixes"
        )

    messages.append({"role": "system", "content": system_str + tools})
    # Iteratively process different roles in the chat
    chat_messages = chat_str_to_messages(example["chat"], tool_to_user=True)
    messages.extend(chat_messages)
    if messages[-1]["role"] != "assistant":
        messages = messages[:-1]  # drop last message if from user

    example["messages"] = messages
    return example


def filter_func(example):
    """
    Simple filter function that returns False if the message list has two consecutive messages
    from the same role. If otherwise, the function returns True.
    """
    messages = example["messages"]
    is_good_entry = 1
    j = 0
    while j + 1 < len(messages):
        if messages[j]["role"] == messages[j + 1]["role"]:
            messages[j]["content"] += messages[j + 1]["content"]
            is_good_entry = 0
            break
        j += 1
    return is_good_entry


def preprocess(ray_ds):
    """
    Preprocesses the Ray dataset into the messages format required by Anyscale Endpoints.
    """
    ray_ds = ray_ds.map(final_mapper)
    ray_ds = ray_ds.filter(filter_func)
    ray_ds = ray_ds.drop_columns(["system", "chat"])  # drop the original columns
    return ray_ds


def pprint_example(example):
    """
    Pretty prints an example with messages and tools field with colors for different roles
    """
    pprint_str = ""
    # ANSI escape code for blue, green, red, yellow and magenta colors
    blue = "\033[94m"
    reset = "\033[0m"
    green = "\033[92m"
    red = "\033[91m"
    magenta = "\033[95m"
    yellow = "\033[93m"
    if "messages" in example:
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
    if "tools" in example:
        pprint_str += f'{magenta}Tool list: {reset}{example["tools"]}\n'
    print(pprint_str)


if __name__ == "__main__":
    import datasets

    hf_ds = datasets.load_dataset(
        "glaiveai/glaive-function-calling-v2", split="train[:10%]"
    )
    # hf_ds = hf_ds.select(range(10))
    # ray_ds = ray.data.from_huggingface(hf_ds)
    # hf_ds = hf_ds.map(mapper)
    # hf_ds = hf_ds.filter(filter_func)
    # hf_ds = hf_ds.remove_columns(["system", "chat"])  # drop the original columns
    # new_ds = preprocess(ray_ds)
    hf_ds = hf_ds.map(final_mapper)
    pprint_example(hf_ds[1])
    import pdb

    pdb.set_trace()
