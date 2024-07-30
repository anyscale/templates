from openai import OpenAI
import base64


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query(base_url: str, api_key: str):

    client = OpenAI(
      base_url=base_url,
      api_key=api_key,
    )

    path = "/mnt/local_storage/kid_drawings/0.JPG"
    chat_completions = client.chat.completions.create(
        model="llava-hf/llava-v1.6-mistral-7b-hf",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Write me a poetry for kid based on this image."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image_to_base64(path)}"}}]}
        ],
        temperature=0.01,
        stream=True
    )

    for chat in chat_completions:
        if chat.choices[0].delta.content is not None:
            print(chat.choices[0].delta.content, end="")

query("http://localhost:8000/v1", "NOT A REAL KEY")
