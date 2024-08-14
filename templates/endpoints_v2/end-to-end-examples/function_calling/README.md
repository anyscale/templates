# Build Tool Calling Feature for LLMs using JSON Mode
**⏱️ Time to complete**: 20 min

**Tool calling** (aka **Function calling**) allows you to hook up external tools to an LLM, enabling it to use APIs and other functions to perform tasks. This feature is particularly useful when you want to extend the capabilities of an LLM beyond its internal knowledge base.

This example explains how to use [JSON Mode](docs.anyscale.com/llms/serving/guides/json_mode) to enable this capability for *any* LLM. 


## How does tool calling work?

Tool calling typically follows three steps:

1. The user submits a query and includes a list of functions along with their parameters and descriptions.

2. The LLM evaluates whether to activate a function:
    * If it decides not to, it responds in natural language– either providing an answer based on its internal knowledge or seeking clarifications about the query and tool usage.
    * If it decides to use a function, it suggests the appropriate API and details on how to employ it, all formatted in JSON.

3. The user then executes the API call in their application and then submits the response back to the LLM. The LLM analyzes the results and continues with any next steps.

Tool calling can be enabled with one of the two following approaches:

1. **Fine-tuning**: You can fine-tune an LLM to use tools when prompted in a specific way. Many recently released open-weight models have gone through some stages of post-training and come out-of-the-box with the capability to use tools. For examples, `mistralai/Mixtral-8x22B-Instruct-v0.1` and `meta-llama/Meta-Llama-3.1-8B-Instruct` natively support tool calling. They have been fine-tuned to do so when given a special prompt format. To use these models, you must specify the tool-compatible prompt format in your LLM config YAML. 

For example, for `mistralai/Mixtral-8x22B-Instruct-v0.1`, here is the prompt format used to enable tool calling (see the full config [here](./llm_configs/mistralai--Mixtral-8x22B-Instruct-v0_1.yaml)):

```yaml
    prompt_format:
        system: "{instruction}\n\n "
        assistant: "{tool_calls}{instruction} </s> "
        # Special part of assistant message (shows the previous assistant message that was a tool call).
        tool_calls: " [TOOL_CALLS] {instruction}"
        # Special new role that should trigger the model to ingest results of tool calls
        tool: "[TOOL_RESULTS] {instruction} [/TOOL_RESULTS]"
        # The format of the available tools that for mixtral goes into the last user message
        tools_list: "[AVAILABLE_TOOLS] {instruction} [/AVAILABLE_TOOLS] "
        trailing_assistant: ""
        user: "{tools_list}[INST] {system}{instruction} [/INST]"
        # Only one BOS is added at the beginning of the entire conversation
        bos: "<s> "
        system_in_user: true
        # Similar to system_in_user, if true it preprends available tools to the user message.
        tools_list_in_user: true
        # If true it will only prepend this message to the last user. 
        system_in_last_user: true
        # If true it will only prepend this message to the last user. 
        tools_list_in_last_user: true
        add_system_tags_even_if_message_is_empty: false
        strip_whitespace: true
        default_system_message: "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity."
```

With this prompt format, RayLLM can provide OpenAI API-compatible tool support for this model. The full config can be generated with the RayLLM CLI (`rayllm gen-config`).

2. **JSON mode**: Not all models come with tool calling capabilities out of the box. In such cases, you can use JSON mode to enable tool calling. JSON mode makes the LLM's output structured and predictable.

In this example, we show how to use JSON mode to build an LLM client that can understand tools in a similar API to the OpenAI SDK.

## Tool calling client

We can break down the problem of tool calling into two stages:

1. Do we need a tool?
2. If yes, how should it be called, and can we use JSON mode to predictably get the response for these questions?

This pseudo code shows the high-level design of a tool-calling client that relies on JSON Mode:

```python
# Predict if any tool should be used
# Uses JSON Mode to predict the tool choice
tool = predict_tool_choice(tools, messages)

if not tool:
    # If no tool is needed, return the normal response
    return normal_response(messages)

# If a tool is needed, predict the tool arguments
# Uses JSON Mode to predict the tool arguments
return return_tool_args(tool, messages)
```

We first use JSON Mode when we want to predict the tool choice so that the response is parsable in a structured way. In this step, we prompt the model to predict whether any tool should be used given the context. For example a prompt like the following is used:


```python
text = "\n\nTo help the user, you are given a set of tools which you can optionally use. Determine which tool to use. If no tool should be used output {\"tool_name\": \"none\"}. Here are the optional list of tools: {tool_list_str}\n\nRules:\n\t1. Output the name of the tool you want to use via {\"tool_name\": \"<name_of_the_tool>\"}.\n\t2. If there has been an error from the previous tool call, output the same tool_name.\n\t3. Output {\"tool_name\": \"none\"} in the following cases:\n\t\t1) Based on context there is ambiguity in what arguments to use for the tools\n\t\t2) The tools are irrelevant for  answering the user's question\n\t\t3) The question is answered based on the previous tool response.\n\t If none of these cases are true, output the name of the tool. Output in JSON."
print(text)
```

    
    
    To help the user, you are given a set of tools which you can optionally use. Determine which tool to use. If no tool should be used output {"tool_name": "none"}. Here are the optional list of tools: {tool_list_str}
    
    Rules:
    	1. Output the name of the tool you want to use via {"tool_name": "<name_of_the_tool>"}.
    	2. If there has been an error from the previous tool call, output the same tool_name.
    	3. Output {"tool_name": "none"} in the following cases:
    		1) Based on context there is ambiguity in what arguments to use for the tools
    		2) The tools are irrelevant for  answering the user's question
    		3) The question is answered based on the previous tool response.
    	 If none of these cases are true, output the name of the tool. Output in JSON.


With a schema that looks like this:

```json
{
    "type": "object",
    "properties": {
        "tool_name": {
            "type": "string",
            "enum": ["get_weather", "add"] + ["none"],
            "description": "The name of the tool to call",
        }
    },
    "required": ["tool_name"],
}
```

This forces the output to have a structure like this:

```json
{"tool_name": "get_weather" | "none"}
```


After the tool has been selected in a separate call, we can ask for the arguments of the tool to be predicted. For example, a prompt like the following is used:


```python
text = "\n\nTo help answer, you have access to the following tool:\n{tool}.\n\n Based on the provided context what should be the arguments to call the tool. If there are errors with the previous tool call, fix the arguments. Output in JSON format."
print(text)
```

    
    
    To help answer, you have access to the following tool:
    {tool}.
    
     Based on the provided context what should be the arguments to call the tool. If there are errors with the previous tool call, fix the arguments. Output in JSON format.


We then use the corresponding tool's schema to guarantee that the output matches the desired schema of the tool. An example of this implementation is done in this [client module](./client/client.py).

## Example usage

The pre-requisite for the following code is to have a self-deployed model (e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`). To do so, you can follow the instructions in the [RayLLM documentation](https://docs.anyscale.com/llms/serving/intro).

Alternatively, you can do `serve run serve_llama_3p1.yaml --non-blocking` to deploy a service locally on a 1xA10 GPU. Make sure to add your HuggingFace token to the [LLM config](./llm_configs/meta-llama--Meta-Llama-3_1-8B-Instruct.yaml) first.


```python
from client import FunctionCallingClient

# This will work if you have deployed the server locally in a workspace.
client = FunctionCallingClient(
    base_url="http://localhost:8000/v1", # PUT YOUR URL HERE
    api_key="fake" # PUT YOUR API KEY HERE
)
```


```python
# Introduce two functions to the model
tools = [
    {
        "type": "function",
        "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
        }
    },
    {
        "type": "function",
        "function": {
        "name": "add",
        "description": "Adds two numbers together",
        "parameters": {
            "type": "object",
            "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
        }
    }
]

messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]


completion = client.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    stream=True,
)

for chunk in completion:
    text = chunk.choices[0].delta.content
    if text:
        print(text, end="")
```

    {"function": {"name": "get_current_weather", "arguments": {"location": "Boston", "unit": "fahrenheit"}}} 

Now that we have got the function, we can assume that we have called the `get_current_weather` tool and received the following response:

```json
{
    "temperature": 25,
    "weather": "sunny"
}
```

We now call the LLM again, returning the response we got with a role of `tool`.


```python
messages.append({'role': 'assistant', 'content': '{"function": {"name": "get_current_weather", "arguments": {"location": "Boston", "unit": "fahrenheit"}}} '})
messages.append({"role": "tool", "content": "{'temperature': 25, 'weather': 'sunny'}"})

completion = client.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    stream=True,
)


for chunk in completion:
    text = chunk.choices[0].delta.content
    if text:
        print(text, end="")
```

    Today in Boston, the temperature is 25 degrees and it is sunny.

## Conclusion

In this example, we have shown how you can use JSON Mode to create a simple client that is capable of tool calling on Any Open-weight LLM, even if the LLM is not fine-tuned to natively support tool calling. 

The implementation in [client.py](./client/client.py) is complete and you can read the code to cover all the details.


