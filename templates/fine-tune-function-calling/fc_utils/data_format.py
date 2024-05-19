"""
Holds constants and data formats used in pre/post processing of the Glaive's function calling dataset.
"""

from enum import Enum
from typing import NamedTuple, Dict, Union

# define our custom type for a tool call
ToolCallType = Dict[str, Union[str, Dict[str, str]]]
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
