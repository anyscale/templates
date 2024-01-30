from openai import OpenAI

# Note: not all arguments are currently supported and will be ignored by the backend.
client = OpenAI(
    base_url="http://localhost:8000/v1", # <- replace http://localhost:8000 with the Anyscale Service query url
    api_key="NOT A REAL KEY", # <- replace with the bearer token if querying an Anyscale Service
)

# Define the messages
messages = [
    {"role": "system", "content": "You are helpful assistant."},
    {"role": "user", "content": "What's the weather like in San Francisco?"}
]
# Define the tools
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
        },
    }
]


response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # auto is default, but we'll be explicit
)
print(response.choices[0].message)
