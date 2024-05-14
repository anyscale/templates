"""
Function call json extraction utils.
Reference: https://gist.github.com/kouroshHakha/6cfbe2bf4aaafc5db733d408044f9902#file-create_test_dataset-py
"""

import json
import re


class FunctionCallNotFoundError(Exception):
    pass


# TODO: Can make this also detect ASSISTANT: <|endoftext|> ASSISTANT: <functioncall> responses and not filter them out in the beginning
def get_tool_call_from_response(assistant_content, eos_present=True):
    assistant_content = assistant_content.strip()  # remove trailing whitespaces
    if assistant_content.startswith("<functioncall>"):
        fn_call_pattern = r"<functioncall>([\s\S]*)" + (
            r"<\|endoftext\|>" if eos_present else ""
        )
        extract_content = False
    else:
        fn_call_pattern = r"([\s\S]*?)<functioncall>([\s\S]*)"
        fn_call_pattern += r"<\|endoftext\|>" if eos_present else ""
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
        tool_call = json.loads(function_call_str)
    else:
        raise FunctionCallNotFoundError("No function call found in assistant response")

    return assistant_content, tool_call


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
