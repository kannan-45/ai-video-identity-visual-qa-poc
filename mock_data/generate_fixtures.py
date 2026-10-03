from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter


BASE_DIR = Path(__file__).resolve().parent

ADA_REFERENCE = BASE_DIR / "inputs" / "adaface" / "reference" / "reference_adaface.jpeg"
GENERATED_DIR = BASE_DIR / "inputs" / "adaface" / "generated"


def generate_fixtures():
    if not ADA_REFERENCE.exists():
        raise FileNotFoundError(
            f"Reference image not found: {ADA_REFERENCE}"
        )

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    reference = Image.open(ADA_REFERENCE).convert("RGB")

    # 1. Identical image
    reference.save(
        GENERATED_DIR / "generated_identical.jpeg",
        quality=95
    )

    # 2. Brightness change
    brightness = ImageEnhance.Brightness(reference).enhance(1.25)
    brightness.save(
        GENERATED_DIR / "generated_brightness.jpeg",
        quality=95
    )

    # 3. Moderate blur
    moderate_blur = reference.filter(
        ImageFilter.GaussianBlur(radius=5)
    )
    moderate_blur.save(
        GENERATED_DIR / "generated_moderate_blur.jpeg",
        quality=95
    )

    # 4. Heavy blur
    heavy_blur = reference.filter(
        ImageFilter.GaussianBlur(radius=18)
    )
    heavy_blur.save(
        GENERATED_DIR / "generated_heavy_blur.jpeg",
        quality=95
    )

    # 5. Crop change
    width, height = reference.size

    crop_x = int(width * 0.12)
    crop_y = int(height * 0.12)

    cropped = reference.crop(
        (
            crop_x,
            crop_y,
            width - crop_x,
            height - crop_y
        )
    )

    cropped = cropped.resize(
        (width, height)
    )

    cropped.save(
        GENERATED_DIR / "generated_crop.jpeg",
        quality=95
    )

    print("Fixture generation completed.")
    print(f"Output directory: {GENERATED_DIR}")

    for file in sorted(GENERATED_DIR.glob("generated_*.jpeg")):
        print(f"  Created: {file.name}")


if __name__ == "__main__":
    generate_fixtures()