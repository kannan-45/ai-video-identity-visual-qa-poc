from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np


def calculate_ssim(reference_path, generated_path):
    reference = Image.open(reference_path).convert("L")
    generated = Image.open(generated_path).convert("L")

    # Resize generated image to reference dimensions
    generated = generated.resize(reference.size)

    reference = np.array(reference)
    generated = np.array(generated)

    score = ssim(reference, generated)

    return round(float(score), 4)


if __name__ == "__main__":
    reference = "mock_data/inputs/reference/reference_001.png"
    generated = "mock_data/inputs/generated/generated_001_same_identity.png"

    score = calculate_ssim(reference, generated)

    print("SSIM score:", score)