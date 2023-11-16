from openai import OpenAI


# Use "meta-llama/Llama-2-7b-chat-hf" to see the result of base model.
# Use "lora-finetuned" or the model id of your choice to see the result of static LoRA models.
#
# The dynamic LoRA model id should be in the format of {base_model_id}:{suffix}:{id}.
# It also expects the checkpoint to be stored in the following path:
#   {base_path}/{base_model_id}:{suffix}:{id}
#   e.g. s3://my-bucket/my-lora-checkouts/meta-llama/Llama-2-7b-chat-hf:lora-model:1234
model = "meta-llama/Llama-2-7b-chat-hf:lora-model:1234"
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

# Note: not all arguments are currently supported and will be ignored by the backend.
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="NOT A REAL KEY",
)
chat_completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "system", "content": system}, {"role": "user", "content": message}],
    temperature=0.01,
)
print(chat_completion)
