import re
import json

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

def extract_functions(system_str):
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
        tools = "["+ ",".join([json.dumps(tool) for tool in tools]) + "]"  # convert to string
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


def combine_multiple_entries(assistant_content):
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



def chat_str_to_messages(chat, tool_to_user=False):
    tag_pattern = re.compile(
        r"(?:USER:\s*(?P<user>.*?)\s*(?=ASSISTANT|$)|ASSISTANT:\s*(?P<assistant>.*?)(?=\n\n\nFUNCTION RESPONSE|\n*(?=USER|$))|\n\n\nFUNCTION RESPONSE:\s*(?P<function_response>.*?)\s*(?=ASSISTANT|USER|$))",
        re.DOTALL,
    )

    matches = tag_pattern.finditer(chat)
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
            assistant_content = combine_multiple_entries(assistant_content)
            if GLAIVEAI_TEMPLATE["tool_call_prefix"] in assistant_content:
                # make function call a list and add tags
                assistant_content = assistant_content.replace(
                    GLAIVEAI_TEMPLATE["tool_call_prefix"], TOOL_CALL_TAGS[0] + " " + "["
                )
                assistant_content += "]" + " " + TOOL_CALL_TAGS[1]
            assistant_content = assistant_content.replace(GLAIVEAI_TEMPLATE["eos"], "")
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
        tools= "["+ ",".join([json.dumps(tool) for tool in tools]) + "]"  # convert to string
        tools = (
            TOOL_LIST_TAGS[0] + " " + tools + " " + TOOL_LIST_TAGS[1]
        )  # add tags
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


def filter_func(example):
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


def preprocess(ray_ds):
    """
    Preprocesses the Ray dataset into the messages format required by Anyscale Endpoints.
    """
    ray_ds = ray_ds.map(final_mapper)
    ray_ds = ray_ds.filter(filter_func)
    ray_ds = ray_ds.drop_columns(["system", "chat"])  # drop the original columns
    return ray_ds


def pprint_example(example,keys=[]):
    """
    Pretty prints an example with messages and tools field with colors for different roles
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


def save_to_jsonl(ds, filepath):
    df = ds.to_pandas()
    df.to_json(filepath, orient="records", lines=True)


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
    test_string = """Sure, I can help you with that. Let me search for books by George Orwell. \n\n\nASSISTANT: <functioncall> [ {\"name\": \"search_books\", \"arguments\": '{\"query\": \"\", \"author\": \"George Orwell\"}'}]"""
    print(combine_multiple_entries(test_string))
    breakpoint()
