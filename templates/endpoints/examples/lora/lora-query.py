"""
A script to query a model with a LoRA adapter spawned with lora-serve.yaml.
The script causes the endpoint to dynamically load a model from the `dynamic_lora_loading_path` set in lora-serve.yaml.
You can also query the base model.

The LoRA adapter should be at an accessible path with the following syntax:
{base_path}/{base_model_id}:{suffix}:{id}
e.g. s3://my-bucket/my-lora-checkouts/meta-llama/Llama-2-7b-chat-hf:lora-model:1234

The model ID used to query the endpoint should have the following syntax:
{base_model_id}:{suffix}:{id}
e.g. meta-llama/Llama-2-7b-chat-hf:lora-model:1234
"""

from openai import OpenAI

# Example LoRA adapter that has been fine-tuned on the viggo dataset (https://arxiv.org/abs/1910.12129)
MODEL = "mistralai/Mistral-7B-Instruct-v0.1:viggo-lora-model:1234"
# You can also query the base model! The output will not be structured according to the viggo dataset.
# MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

# We first contstruct our messages according to the viggo dataset
system = ("Given a target sentence construct the underlying meaning representation\n"
          "of the input sentence as a single function with attributes and attribute\n"
          "values. This function should describe the target string accurately and the\n"
          "function must be one of the following ['inform', 'request', 'give_opinion',\n"
          "'confirm', 'verify_attribute', 'suggest', 'request_explanation',\n"
          "'recommend', 'request_attribute'].\n\n"
          "The attributes must be one of the following:\n"
          "['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating',\n"
          "'genres', 'player_perspective', 'has_multiplayer', 'platforms',\n"
          "'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']\n")
message = ("Here is the target sentence:\n"
           "Dirt: Showdown is a sport racing game that was released in 2012. The game"
           " is available on PlayStation, Xbox, and PC, and it has an ESRB Rating of E"
           " 10+ (for Everyone 10 and Older). However, it is not yet available as a"
           " Steam, Linux, or Mac release.")

# Note: Not all arguments of the OpenAI API are currently supported and will be ignored by the backend.
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="NOT A REAL KEY",
)
chat_completion = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "system", "content": system}, {"role": "user", "content": message}],
    temperature=0.01,
)

print("Answer of LoRA-model (structured according to the viggo dataset (https://arxiv.org/abs/1910.12129)):\n")
print(chat_completion.choices[0].message.content)
