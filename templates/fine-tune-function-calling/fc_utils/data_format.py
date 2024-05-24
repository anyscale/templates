"""
Holds constants and data format utils used in pre/post-processing of GlaveAI's function calling dataset
"""

from enum import Enum
from typing import NamedTuple, Dict, Union, Any, List
from dataclasses import dataclass

GLAIVEAI_SYSTEM_NO_TOOLS = (
    "SYSTEM: You are a helpful assistant, with no access to external functions."
)
GLAIVEAI_SYSTEM_WITH_TOOLS = "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -"
GLAIVEAI_TOOL_CALL_PREFIX = "<functioncall>"
GLAIVEAI_EOS = "<|endoftext|>"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# System prompts for the base model.
BASE_MODEL_WITH_FUNCTIONS_SYSTEM_PROMPT = """You are a helpful assistant with access to a set of possible tools/functions. Calling these tools is optional. Your goal is to best answer the user's queries. Carefully consider each query and determine if one or more tools can be used to answer the query. If none of the tools can be used, point it out. If the given question lacks the parameters strictly required by the tool, point it out and request the user for the required details. You should place all of your tool calls in a list, enclosed by special tags - [TOOL_CALLS] and [/TOOL_CALLS]. Only then will it be valid!

If you wish to call a particular tool, specify the name of the tool and any arguments in a format that conforms to the underlying function's schema. Tool calls should be in this format: [TOOL_CALLS] [{{"type" : "function", "function": {{"name": "func1": "arguments": "{{\\\"param1\\\": \\\"value\\\", \\\"param2\\\": \\\"value2\\\"}}"}}}}] [/TOOL_CALLS]. Note that the list of calls is enclosed in square brackets. There are double quotes around the function name and arguments, and the "arguments" entry is a strigified JSON. Each double quote inside the arguments value is double escaped.

After making one or more tool calls, do not respond further. You will receive tool results between the special indicators [TOOL_RESULT] and [/TOOL_RESULT]. You are not meant to output or hallucinate a tool result! Only after you get the required result(s) from the tool(s), you should provide a complete answer to the user's query.

Here are the tools you can use {tools}."""
BASE_MODEL_NO_FUNCTIONS_SYSTEM_PROMPT = (
    "You are a helpful assistant with no access to external functions."
)


# Define our custom type for tool call and messages
ToolCallType = Dict[str, Union[str, Dict[str, Any]]]
MessageType = Dict[str, Any]


class IndicatorTags(NamedTuple):
    start: str
    end: str

    def __repr__(self):
        return f"{self.start} ... {self.end}"


class GlaiveAIRoleTags(Enum):
    USER = "USER:"
    ASSISTANT = "ASSISTANT:"
    TOOL = "FUNCTION RESPONSE:"


class DatasetFormat(Enum):
    GLAIVE = "glaive"
    ANYSCALE = "anyscale"
    OPENAI = "openai"


TOOL_CALL_TAGS = IndicatorTags(start="[TOOL_CALLS]", end="[/TOOL_CALLS]")
TOOL_RESULT_TAGS = IndicatorTags(start="[TOOL_RESULT]", end="[/TOOL_RESULT]")
TOOL_LIST_TAGS = IndicatorTags(start="[TOOL_LIST]", end="[/TOOL_LIST]")
# GlaiveAI's function calls are in the format "<functioncall> {...} <|endoftext|>
GLAIVEAI_TOOL_CALL_INDICATORS = IndicatorTags(GLAIVEAI_TOOL_CALL_PREFIX, GLAIVEAI_EOS)


# TODO: Put all tool call dicts in this Function call dataclass for easy access
@dataclass
class FunctionCall:
    name: str
    arguments: Dict[str, str]


def _check_tool_call_format(
    tool_call: Dict[str, Any], format: DatasetFormat = DatasetFormat.OPENAI
) -> bool:
    """Checks if the tool call is in the correct format."""
    if format == DatasetFormat.GLAIVE:
        # In Glaive, the tool call uses the older "function_call" OpenAI format.
        # First, we bring it into the new OpenAI format
        tool_call = {"type": "function", "function": tool_call}
    # Check if the tool call is a function call. Only function calls are supported.
    if "type" not in tool_call or tool_call["type"] != "function":
        return False
    if "function" not in tool_call:
        return False
    function_call = tool_call["function"]
    # Tool call should have a "name" and "arguments" field
    if "name" not in function_call or "arguments" not in function_call:
        return False
    # "arguments" entry should be a dictionary and "name" should be a string
    elif not (
        isinstance(function_call["arguments"], dict)
        and isinstance(function_call["name"], str)
    ):
        return False
    return True


def check_tool_calls_format(
    tool_calls: List[Dict[str, Any]], format: DatasetFormat = DatasetFormat.OPENAI
) -> bool:
    """Checks if all tool calls in a list of tool calls the correct format."""
    if not isinstance(tool_calls, list):
        return ValueError("Tool calls should be a list.")
    for tool_call in tool_calls:
        # Check if the tool call is a function call. Only function calls are supported.
        if not _check_tool_call_format(tool_call, format):
            return False
    return True
