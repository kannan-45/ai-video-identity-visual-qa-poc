import torch
import lpips
from PIL import Image
from torchvision import transforms


def load_image(path):
    image = Image.open(path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    return transform(image).unsqueeze(0)


loss_fn = lpips.LPIPS(net="alex")

reference = load_image(
    "mock_data/inputs/reference/reference_001.png"
)

generated = load_image(
    "mock_data/inputs/generated/generated_002_blur.png"
)

with torch.no_grad():
    score = loss_fn(reference, generated)

print("LPIPS score:", round(float(score.item()), 4))