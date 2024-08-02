
from typing import Optional, List, Literal, Dict, Any

from pydantic import BaseModel


class FunctionCall(BaseModel):
    name: str
    arguments: Optional[str] = None

class ToolCall(BaseModel):
    function: FunctionCall
    type: Literal["function"]
    id: str

class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class Message(BaseModel):
    role: Literal["system", "assistant", "user", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ToolChoice(BaseModel):
    type: Literal["function"]
    function: Function


class Tool(BaseModel):
    type: Literal["function"]
    function: Function
