from typing import Optional, List, Dict, Any

import openai
import copy
import json

from .models import Message, Tool


TOOL_RESULTS_PROMPT = "Ok. Here is the response from the previous tool call:\n {tool_response}\n Given, this context, what is the answer to the user?. Do not provide any extra explaination, just output the answer based on the returned response from the tool call."

TOOL_LIST_SYSTEM_PROMPT = "{input_msg_system}\n\nTo help the user, you are given a set of tools which you can optionally use. Determine which tool to use. If no tool should be used output {no_tool_str}. Here are the optional list of tools: {tool_list_str}\n\nRules:\n\t1. Output the name of the tool you want to use via {tool_use_str}.\n\t2. If there has been an error from the previous tool call, output the same tool_name.\n\t3. Output {no_tool_str} in the following cases:\n\t\t1) Based on context there is ambiguity in what arguments to use for the tools\n\t\t2) The tools are irrelevant for  answering the user's question\n\t\t3) The question is answered based on the previous tool response.\n\t If none of these cases are true, output the name of the tool. Output in JSON."

EXTRA_USER_PROMPT = "Based on this context determine the correct tool_name. Remember to not output {no_tool_str} if the previous tool response indicates a system error and output it if the previous tool response answers the question."


TOOL_ARGS_USER_PROMPT = "\n\nTo help answer, you have access to the following tool:\n{tool}.\n\n Based on the provided context what should be the arguments to call the tool. If there are errors with the previous tool call, fix the arguments. Output in JSON format."


class FunctionCallingClient:

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> None:
        
        self._client = openai.Client(
            base_url=base_url,
            api_key=api_key
        )
        
        
    def _prepare_tool_choice_msgs(self, tools: List[Tool], messages: List[Message]) -> List[Message]:
        
        msgs = copy.deepcopy(messages)
        
        tool_strs = [
            json.dumps(
                {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    # NOTE: Adding parameters here provides more info about parameter default values. so if the user request given parameters is ambiguous, the assistant can ask for clarification.
                    "parameters": tool.function.parameters,
                }
            )
            for tool in tools
        ]
        
        tool_list_str = "\n\n" + "\n\n".join(tool_strs)
        no_tool_str = json.dumps({"tool_name": "none"})
        tool_use_str = str({"tool_name": "<name_of_the_tool>"})
        
        input_msg_system = ""
        if msgs[0].role == "system":
            input_msg_system = msgs[0].content or ""
            msgs.pop(0)

            
        if msgs[-1].role != "user":
            raise ValueError("The last message must be from the user")
        
        # Make sure the last message has content
        msgs[-1].content = msgs[-1].content or ""
            
        system_msg = TOOL_LIST_SYSTEM_PROMPT.format(
            input_msg_system=input_msg_system,
            tool_list_str=tool_list_str,
            no_tool_str=no_tool_str,
            tool_use_str=tool_use_str
        )
        msgs.insert(0, Message(role="system", content=system_msg))
        
        msgs[-1].content += "\n\n" + EXTRA_USER_PROMPT.format(no_tool_str=no_tool_str)
        
        return msgs
        
    
    def _predict_tool_choice(self, model: str, tools: List[Tool], messages: List[Message]) -> Optional[Tool]:
        
        tool_names = [tool.function.name for tool in tools]
        
        tool_choice_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "enum": tool_names + ["none"],
                        "description": "The name of the tool to call",
                    }
                },
                "required": ["tool_name"],
            }
        )
        
        tool_choice_messages = self._prepare_tool_choice_msgs(tools, messages)
        tool_choice_messages = [msg.model_dump(exclude_none=True) for msg in tool_choice_messages]
        
        completion = self._client.chat.completions.create(
            model=model,
            messages=tool_choice_messages,
            response_format={
                "type": "json_object",
                "schema": tool_choice_schema,
            },
        )
        
        returned_msg = completion.choices[-1].message.content or ""
        try:
            tool_name = json.loads(returned_msg)["tool_name"]
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding tool choice response: {returned_msg}")
        
        if tool_name == "none":
            return None

        selected_tool = next(tool for tool in tools if tool.function.name == tool_name)
        return selected_tool
    
    
    
    def _replace_tool_assistant_message(self, messages: List[Message]) -> List[Message]:
        msgs = copy.deepcopy(messages)
        
        for msg in msgs:
            
            if msg.role == "tool":
                msg.role = "user"
                msg.content = TOOL_RESULTS_PROMPT.format(
                    tool_response=msg.content or "",
                )

        return msgs
        
        
    def _prepare_tool_args_msg(self, tool: Tool, messages: List[Message]) -> List[Message]:
        msgs = copy.deepcopy(messages)
        
        extra_usr_msg = TOOL_ARGS_USER_PROMPT.format(tool=tool.function.model_dump_json())
        if isinstance(msgs[-1].content, str):
            msgs[-1].content = (
                (msgs[-1].content or "") + " " + extra_usr_msg
            )
        
        return msgs

    def _return_tool_arguments(self, messages: List[Message], selected_tool: Tool, **kwargs) -> Dict[str, Any]:
        
        
        tool_name = selected_tool.function.name
        arguments_schema = selected_tool.function.parameters
        
        tool_arg_messages = self._prepare_tool_args_msg(selected_tool, messages)
        
        tool_arg_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "enum": [tool_name],
                            },
                            "arguments": arguments_schema,
                        },
                        "required": ["name", "arguments"],
                    }
                },
                "required": ["function"],
            }
        )
        
        completion = self._client.chat.completions.create(
            messages=tool_arg_messages,
            response_format={
                "type": "json_object",
                "schema": tool_arg_schema,
            },
            **kwargs,
        )
        
        return completion
        


    def create(self, **kwargs) -> Dict[str, Any]:
        tools = kwargs.pop("tools")
        tool_choice = kwargs.pop("tool_choice")
        
        if tool_choice not in ["none", "auto"]:
            raise ValueError("tool_choice must be 'none' or 'auto'")

        if not tools or tool_choice == "none":
            return self._client.chat.completions.create(**kwargs)
        
        messages = [
            Message(**msg) for msg in 
            kwargs.pop("messages", [])
        ]
        
        # Replace the tool messages with user / assistant 
        # equivalents.
        tools = [Tool(**tool) for tool in tools]

        modified_messages = self._replace_tool_assistant_message(messages)
        
        model = kwargs.get("model", "")
        selected_tool = self._predict_tool_choice(
            model=model, tools=tools, messages=modified_messages)
        
        if selected_tool is None:
            return self._client.chat.completions.create(**kwargs, messages=modified_messages)

        return self._return_tool_arguments(modified_messages, selected_tool, **kwargs)

    