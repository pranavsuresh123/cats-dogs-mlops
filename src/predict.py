from pathlib import Path

import torch
from PIL import Image

from src.model import CatsDogsCNN
from src.preprocess import get_eval_transform


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = Path("models/model.pt")


class Predictor:

    def __init__(self):
        self.model = CatsDogsCNN().to(DEVICE)

        self.model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=DEVICE
            )
        )

        self.model.eval()

        self.transform = get_eval_transform()

    def predict(self, image: Image.Image):

        image = image.convert("RGB")

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            output = self.model(tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]

        cat_probability = probabilities[0].item()
        dog_probability = probabilities[1].item()

        predicted_class = (
            "cat"
            if cat_probability >= dog_probability
            else "dog"
        )

        return {
            "class": predicted_class,
            "cat_probability": round(
                cat_probability,
                4
            ),
            "dog_probability": round(
                dog_probability,
                4
            ),
            "probability": round(
                max(
                    cat_probability,
                    dog_probability
                ),
                4
            )
        }