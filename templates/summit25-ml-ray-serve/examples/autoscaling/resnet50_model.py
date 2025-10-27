import requests
from io import BytesIO

from PIL import Image
import starlette.requests
from ray import serve


@serve.deployment
class Model:
    def __init__(self):
        from torchvision import transforms
        import torchvision.models as models
        from torchvision.models import ResNet50_Weights

        self.resnet50 = (
            models.resnet50(weights=ResNet50_Weights.DEFAULT).eval().to("cpu")
        )
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        resp = requests.get(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        self.categories = resp.content.decode("utf-8").split("\n")

    async def __call__(self, request: starlette.requests.Request) -> str:
        import torch

        uri = (await request.json())["uri"]
        image_bytes = requests.get(uri).content
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        input_tensor = torch.cat([self.preprocess(image).unsqueeze(0)]).to("cpu")
        with torch.no_grad():
            output = self.resnet50(input_tensor)
            sm_output = torch.nn.functional.softmax(output[0], dim=0)
        ind = torch.argmax(sm_output)
        return self.categories[ind]


app = Model.bind()
