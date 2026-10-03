from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np

reference = np.array(
    Image.open("mock_data/inputs/reference/reference_001.png")
    .convert("L")
)

generated = np.array(
    Image.open("mock_data/inputs/generated/generated_002_blur.png")
    .convert("L")
)

score = ssim(reference, generated)

print("SSIM score:", round(score, 4))

if score >= 0.90:
    print("Result: VISUALLY CONSISTENT")
else:
    print("Result: VISUAL DRIFT")