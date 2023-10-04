import openai

# import aviary.backend.router_application

# Note: not all arguments are currently supported and will be ignored by the backend.
chat_completion = openai.ChatCompletion.create(
    api_base="http://localhost:8000/v1", api_key="",
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What are some of the highest rated restaurants in San Francisco?'."}],
    temperature=0.01
)
print(chat_completion)