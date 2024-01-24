from openai import OpenAI


# Note: not all arguments are currently supported and will be ignored by the backend.
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="NOT A REAL KEY",
)
chat_completion = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What are some of the highest rated restaurants in San Francisco?'."}],
    temperature=0.01
)
print(chat_completion)
