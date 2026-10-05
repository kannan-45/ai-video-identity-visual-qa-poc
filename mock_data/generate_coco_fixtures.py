from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COCO_DIR = PROJECT_ROOT / "mock_data" / "inputs" / "coco"
IMAGE_DIR = COCO_DIR / "images"
FAILURE_DIR = COCO_DIR / "failures"


def create_fixtures():
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)

    source = IMAGE_DIR / "000000000139.jpg"
    replacement = IMAGE_DIR / "000000000285.jpg"

    if not source.exists():
        raise FileNotFoundError(f"COCO image not found: {source}")

    if not replacement.exists():
        raise FileNotFoundError(f"COCO image not found: {replacement}")

    image = Image.open(source).convert("RGB")
    replacement_image = Image.open(replacement).convert("RGB")

    width, height = image.size

    replacement_image = replacement_image.resize(
        (width, height)
    )

    # --------------------------------------------------
    # Blur failure
    # --------------------------------------------------

    blurred = image.filter(
        ImageFilter.GaussianBlur(radius=12)
    )

    blurred.save(
        FAILURE_DIR / "000000000139_blur.jpg",
        quality=95
    )

    # --------------------------------------------------
    # Brightness failure
    # --------------------------------------------------

    darkened = ImageEnhance.Brightness(
        image
    ).enhance(0.25)

    darkened.save(
        FAILURE_DIR / "000000000139_dark.jpg",
        quality=95
    )

    # --------------------------------------------------
    # Crop failure
    # --------------------------------------------------

    left = int(width * 0.25)
    top = int(height * 0.25)
    right = int(width * 0.75)
    bottom = int(height * 0.75)

    cropped = image.crop(
        (left, top, right, bottom)
    )

    cropped = cropped.resize(
        (width, height)
    )

    cropped.save(
        FAILURE_DIR / "000000000139_crop.jpg",
        quality=95
    )

    # --------------------------------------------------
    # Object replacement failure
    # --------------------------------------------------

    object_changed = image.copy()

    left = int(width * 0.20)
    top = int(height * 0.20)
    right = int(width * 0.80)
    bottom = int(height * 0.80)

    replacement_crop = replacement_image.crop(
        (
            left,
            top,
            right,
            bottom
        )
    )

    object_changed.paste(
        replacement_crop,
        (left, top)
    )

    object_changed.save(
        FAILURE_DIR / "000000000139_object_change.jpg",
        quality=95
    )

    print("COCO fixtures created successfully.")


if __name__ == "__main__":
    create_fixtures()