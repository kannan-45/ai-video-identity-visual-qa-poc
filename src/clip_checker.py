import torch
import open_clip
from PIL import Image


class CLIPChecker:

    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai"
        )

        self.model.eval()

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        return self.preprocess(image).unsqueeze(0)

    def calculate(self, reference_path, generated_path):

        reference = self._load_image(reference_path)
        generated = self._load_image(generated_path)

        with torch.no_grad():

            reference_embedding = self.model.encode_image(reference)
            generated_embedding = self.model.encode_image(generated)

            reference_embedding /= reference_embedding.norm(dim=-1, keepdim=True)
            generated_embedding /= generated_embedding.norm(dim=-1, keepdim=True)

            similarity = (
                reference_embedding @ generated_embedding.T
            ).item()

        return round(float(similarity), 4)


if __name__ == "__main__":

    checker = CLIPChecker()

    score = checker.calculate(
        "mock_data/inputs/reference/reference_001.png",
        "mock_data/inputs/generated/generated_001_same_identity.png"
    )

    print("CLIP similarity:", score)