import requests

HOST = "http://localhost:8000" # Replace with Anyscale Service URL for non-local deployment
TOKEN = "TODO_INSERT_YOUR_SERVICE_TOKEN" # Replace with Anyscale Service Bearer token for non-local deployment

def generate_image(prompt, image_size):
    req = {"prompt": prompt, "img_size": image_size}
    response: requests.Response = requests.get(
        f"{HOST}/imagine",
        params={"prompt": prompt, "img_size": image_size},
        headers={
            "Authorization": f"Bearer {TOKEN}",
        },
    )
    return response.content


image = generate_image("twin peaks sf in basquiat painting style", 640)
filename = "image.png"
with open(filename, "wb") as f:
    f.write(image)
