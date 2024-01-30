from openai import OpenAI


# Note: not all arguments are currently supported and will be ignored by the backend.
client = OpenAI(
    base_url="http://localhost:8000/v1", # <- replace http://localhost:8000 with the Anyscale Service query url
    api_key="NOT A REAL KEY", # <- replace with the bearer token if querying an Anyscale Service
)
chat_completion = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What are some of the highest rated restaurants in San Francisco?'."}],
    temperature=0.01
)
print(chat_completion)
