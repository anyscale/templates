"""
Response parsers for Anyscale Endpoints and OpenAI models. We define two parser classes `AnyscaleResponseParser` and
`OpenAIResponseParser` below to send messages to the respective endpoints and parse the result.
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import openai
from openai import OpenAI

from fc_utils.function_extraction_utils import (
    FunctionCallFormatError,
    get_tool_calls_from_response,
)
from fc_utils.data_format import IndicatorTags, DatasetFormat, ToolCallType

NUM_RETRIES = 5
SLEEP_INTERVAL_BETWEEN_RETRIES = 10
ERROR_OUTPUT = "$$RUNTIME_ERROR$$"
INCORRECT_FORMAT = "$$INCORRECT_FORMAT$$"


# Dataclass to store the parsed response
@dataclass
class ParsedResponse:
    content: str
    tool_calls: List[ToolCallType]
    original_response: "ChatCompletionMessage"


def get_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> "ChatCompletion":
    """
    Gets completion from the OpenAI ChatCompletion API for the provided OpenAI client and model.
    """
    # Simple way to handle rate limit errors with retries
    for _ in range(NUM_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response
        except openai.AuthenticationError as e:
            raise e
        except openai.OpenAIError as e:
            # This will capture all other errors, including rate limit errors and formatting errors
            print(f"Error: {e}")
            time.sleep(SLEEP_INTERVAL_BETWEEN_RETRIES)
    # Error response
    return ERROR_OUTPUT


class ResponseParser(ABC):
    """
    Abstract base class for response parsers.
    """

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        tool_call_tags: Optional[IndicatorTags] = None,
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.tool_call_tags = tool_call_tags

    @abstractmethod
    def get_parsed_response(
        self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]
    ) -> ParsedResponse:
        """
        Gets completion for input messages and returns the processed response. Tool calls, if present
        are extracted and parsed as json.

        Args:
            messages: List of messages to send to the model
            tools: List of tools available to the model

        Returns:
            processed_response: Dict containing the content of the parsed response, tool calls if present, and the original response
        """
        pass


class AnyscaleResponseParser(ResponseParser):
    """Response parser for models hosted on Anyscale endpoints.

    Assumes that the model response has tool calls formatted between tool_call_tags.
    """

    def get_parsed_response(
        self, messages, tools=None, temperature=0.0, max_tokens=256
    ):
        # Tools is ignored as the tool list would be included in the system prompt
        response = get_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Default error output
        processed_response = ParsedResponse(
            content=ERROR_OUTPUT, tool_calls=None, original_response=response
        )
        if response == ERROR_OUTPUT:
            return processed_response

        response_message_content = response.choices[0].message.content
        processed_response.content = response_message_content
        processed_response.original_response = response.choices[0].message

        # Check if the content includes tool call tags
        if (
            response_message_content
            and self.tool_call_tags.start in response_message_content
        ):
            try:
                response_message_content, tool_calls = get_tool_calls_from_response(
                    response_message_content,
                    self.tool_call_tags,
                    format=DatasetFormat.ANYSCALE,
                )
                processed_response.content = response_message_content
                processed_response.tool_calls = tool_calls
            except FunctionCallFormatError:
                # This handles either a function call not being found or the json not being decoded properly.
                # One example for the second case can be missed commas/quotes in the arguments field.
                processed_response.content = response_message_content
                processed_response.tool_calls = INCORRECT_FORMAT

        return processed_response


class OpenAIResponseParser(ResponseParser):
    """Response parser for OpenAI models."""

    def get_parsed_response(self, messages, tools, temperature=0.0, max_tokens=256):
        response = get_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            tools=tools if len(tools) else None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Default error output
        processed_response = ParsedResponse(
            content=ERROR_OUTPUT, tool_calls=None, original_response=response
        )
        if response == ERROR_OUTPUT:
            return processed_response

        response_message = response.choices[0].message
        processed_response.content = response_message.content
        processed_response.original_response = response_message

        tool_calls = response_message.tool_calls
        # Process tool calls if present
        if tool_calls:
            processed_response.tool_calls = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    output_tool = {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": function_args,
                        },
                    }
                    processed_response.tool_calls.append(output_tool)
                except json.JSONDecodeError:
                    processed_response.tool_calls = INCORRECT_FORMAT
                    break
        return processed_response
