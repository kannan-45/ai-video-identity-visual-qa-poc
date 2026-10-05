import torch
import lpips
from PIL import Image
from torchvision import transforms


# Load LPIPS model once
_lpips_model = lpips.LPIPS(net="alex")
_lpips_model.eval()


def _load_image(image_path):
    """
    Load image and convert it into the tensor format
    expected by LPIPS.
    """
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    return transform(image).unsqueeze(0)


def calculate_lpips(reference_path, generated_path):
    """
    Calculate LPIPS perceptual distance.

    Lower score = more visually similar.
    Higher score = greater perceptual difference.
    """

    reference = _load_image(reference_path)
    generated = _load_image(generated_path)

    with torch.no_grad():
        score = _lpips_model(reference, generated)

    return round(float(score.item()), 4)


if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    reference = (
        PROJECT_ROOT
        / "mock_data"
        / "inputs"
        / "adaface"
        / "reference"
        / "reference_adaface.jpeg"
    )

    generated = (
        PROJECT_ROOT
        / "mock_data"
        / "inputs"
        / "adaface"
        / "generated"
        / "generated_visual_drift.jpeg"
    )

    score = calculate_lpips(
        str(reference),
        str(generated)
    )

    print("LPIPS score:", score)