import requests

endpoint = "http://localhost:8000/generate"


def generate_image(prompt, image_size):
    req = {"prompt": prompt}
    resp = requests.post(endpoint, params=req)
    return resp.content

i = 0 
while i < 10:
    i = i + 1
    image = generate_image("a photo of an astronaut riding a horse on mars", 640)
    filename = "image.png"
    with open(filename, "wb") as f:
        f.write(image)
