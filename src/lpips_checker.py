import torch
import lpips
from PIL import Image
from torchvision import transforms


class LPIPSChecker:

    def __init__(self, network="alex"):
        self.model = lpips.LPIPS(net=network)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        return self.transform(image).unsqueeze(0)

    def calculate(self, reference_path, generated_path):
        reference = self._load_image(reference_path)
        generated = self._load_image(generated_path)

        with torch.no_grad():
            score = self.model(reference, generated)

        return round(float(score.item()), 4)


if __name__ == "__main__":
    checker = LPIPSChecker()

    score = checker.calculate(
        "mock_data/inputs/reference/reference_001.png",
        "mock_data/inputs/generated/generated_001_same_identity.png"
    )

    print("LPIPS score:", score)