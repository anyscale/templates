from openai import OpenAI

import gradio as gr

# Sample URL: https://air-example-data-2.s3.amazonaws.com/llava_example_kid_drawings/0.JPG


def write_poetry(url: str):
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="NOT A REAL KEY",
    )
    chat_completions = client.chat.completions.create(
        model="llava-hf/llava-v1.6-mistral-7b-hf",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Write me a poetry for kid based on this image."},
                {"type": "image_url", "image_url": {
                   "url": url}}]}
        ],
        temperature=0.01,
        stream=False
    )

    return chat_completions.choices[0].message.content.strip()

with gr.Blocks() as demo:
    gr.Markdown("Enter image URL.")

    with gr.Row():
        with gr.Column():
            image_url_input = gr.Textbox(label="Enter image URL")
            submit_btn = gr.Button("Submit")
            image_display = gr.Image(type="pil", label="Input Image")

        with gr.Column():
            text_output = gr.Textbox(label="Output Text", lines=10, interactive=False)
    submit_btn.click(lambda x: x, inputs=image_url_input, outputs=image_display)
    submit_btn.click(write_poetry, inputs=image_url_input, outputs=text_output)

demo.launch()
