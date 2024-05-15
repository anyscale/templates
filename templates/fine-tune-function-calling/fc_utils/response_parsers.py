"""
Response parsers for Anyscale Endpoints and OpenAI models. We define two parser classes `AnyscaleResponseParser` and
`OpenAIResponseParser` below to send messages to the respective endpoints and parse the result. With the
`AnyscaleResponseParser`, we use the helper functions defined in `function_extraction_utils` to obtain the
function call json from the response text.
"""
import json
import time

import openai
from openai import OpenAI

from fc_utils.function_extraction_utils import (
    FunctionCallNotFoundError,
    get_tool_calls_from_response,
)

NUM_RETRIES = 5
SLEEP_INTERVAL_BETWEEN_RETRIES = 10
ERROR_OUTPUT = "$$RUNTIME_ERROR$$"


def get_completion(client, model, messages, tools=None):
    """
    Gets completion from the OpenAI ChatCompletion API for the provided OpenAI client and model
    """
    # simple way to handle rate limit errors with retries
    for _ in range(NUM_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, tools=tools
            )
            return response
        except openai.AuthenticationError as e:
            raise e
        except (
            openai.OpenAIError
        ) as e:  # this will capture all other errors, including rate limit errors and formatting errors
            print(f"Error: {e}")
            time.sleep(SLEEP_INTERVAL_BETWEEN_RETRIES)

    return ERROR_OUTPUT  # error response


# TODO: currently, if the response is as follows: "arguments": '{"text": "there is a "double quote" here"}', the json will not be decoded properly.
class AnyscaleResponseParser:
    def __init__(self, api_key, api_base, model):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model

    def get_parsed_response(self, messages):
        """
        Gets completion for input messages and returns the processed response. Tool calls, if present
        are extracted and parsed as json.
        """
        response = get_completion(self.client, self.model, messages)
        if response == ERROR_OUTPUT:
            processed_response = {
                "content": ERROR_OUTPUT,
                "tool_calls": None,
                "original_response": ERROR_OUTPUT,
            }
            return processed_response
        response_message_content = response.choices[0].message.content
        processed_response = {
            "content": response_message_content,
            "tool_calls": None,
            "original_response": response.choices[0].message,
        }
        if response_message_content and "<functioncall>" in response_message_content:
            try:
                response_message_content, tool_calls = get_tool_calls_from_response(response_message_content)
                processed_response["content"] = response_message_content
                processed_response["tool_calls"] = tool_calls
            except (FunctionCallNotFoundError, json.JSONDecodeError):
                # this handles either a function call not being found or the json not being decoded properly.
                # one example for the second case can be missed commas/quotes in the arguments field.
                processed_response["content"] = response_message_content
                processed_response["tool_calls"] = None
        return processed_response


class OpenAIResponseParser:
    def __init__(self, api_key, api_base, model):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model

    def get_parsed_response(self, messages, tools):
        """
        Gets completion for input messages and returns the processed response. Tool calls, if present
        are extracted and parsed as json.
        """
        response = get_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            tools=tools if len(tools) else None,
        )
        if response == ERROR_OUTPUT:
            processed_response = {
                "content": ERROR_OUTPUT,
                "tool_calls": None,
                "original_response": response,
            }
            return processed_response
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        processed_response = {
            "content": response_message.content,
            "tool_calls": None,
            "original_response": response_message,
        }
        # process tool calls if present
        if tool_calls:
            processed_response["tool_calls"] = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    output_function = {"name": function_name, "arguments": function_args}
                    processed_response["tool_calls"].append(output_function)
                except json.JSONDecodeError:
                    continue
        return processed_response
