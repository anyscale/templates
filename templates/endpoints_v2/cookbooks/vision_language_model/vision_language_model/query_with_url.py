from openai import OpenAI


def query(base_url: str, api_key: str):

   client = OpenAI(
     base_url=base_url,
     api_key=api_key,
   )
   chat_completions = client.chat.completions.create(
       model="llava-hf/llava-v1.6-mistral-7b-hf",
       messages=[
           {"role": "user", "content": [
               {"type": "text", "text": "Write me a poetry for kid based on this image."},
               {"type": "image_url", "image_url": {
                   "url": "https://air-example-data-2.s3.amazonaws.com/llava_example_kid_drawings/0.JPG"}}]}
       ],
       temperature=0.01,
       stream=True
   )

   for chat in chat_completions:
       if chat.choices[0].delta.content is not None:
           print(chat.choices[0].delta.content, end="")

query("http://localhost:8000/v1", "NOT A REAL KEY")
