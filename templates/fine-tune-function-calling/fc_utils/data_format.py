"""
Holds constants and data format utilities in pre/post processing of the Glaive's function calling dataset.
"""

from enum import Enum
from typing import NamedTuple, Dict, Union, Any, List
from dataclasses import dataclass

# define our custom type for tool call and messages
ToolCallType = Dict[str, Union[str, Dict[str, Any]]]
MessageType = Dict[str, str]


class IndicatorTags(NamedTuple):
    start: str
    end: str

    def __repr__(self):
        return f"{self.start} ... {self.end}"


class GlaiveAIRoleTags(Enum):
    USER = "USER: "
    ASSISTANT = "ASSISTANT: "
    TOOL = "FUNCTION RESPONSE: "


class DatasetFormat(Enum):
    GLAIVE = "glaive"
    ANYSCALE = "anyscale"
    OPENAI = "openai"


TOOL_CALL_TAGS = IndicatorTags(start="[TOOL_CALLS]", end="[/TOOL_CALLS]")
TOOL_RESULT_TAGS = IndicatorTags(start="[TOOL_RESULT]", end="[/TOOL_RESULT]")
TOOL_LIST_TAGS = IndicatorTags(start="[TOOL_LIST]", end="[/TOOL_LIST]")

GLAIVEAI_SYSTEM_NO_TOOLS = (
    "SYSTEM: You are a helpful assistant, with no access to external functions."
)
GLAIVEAI_SYSTEM_WITH_TOOLS = "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -"
GLAIVEAI_TOOL_CALL_PREFIX = "<functioncall>"
GLAIVEAI_EOS = "<|endoftext|>"
GLAIVEAI_TOOL_CALL_INDICATORS = IndicatorTags(GLAIVEAI_TOOL_CALL_PREFIX, GLAIVEAI_EOS)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# TODO: convert everything to the Function call dataclass for easy access
@dataclass
class FunctionCall:
    name: str
    arguments: Dict[str, str]


def check_tool_call_format(tool_calls: List[ToolCallType]) -> bool:
    """Checks if the tool call is in the correct format."""
    for tool_call in tool_calls:
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
