from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


BASE_DIR = Path(__file__).resolve().parent

ADA_REFERENCE = (
    BASE_DIR
    / "inputs"
    / "adaface"
    / "reference"
    / "reference_adaface.jpeg"
)

GENERATED_DIR = (
    BASE_DIR
    / "inputs"
    / "adaface"
    / "generated"
)


def generate_fixtures():
    if not ADA_REFERENCE.exists():
        raise FileNotFoundError(
            f"Reference image not found: {ADA_REFERENCE}"
        )

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    reference = Image.open(ADA_REFERENCE).convert("RGB")

    # ---------------------------------------------------------
    # 1. Identical image
    # Expected: SAME_IDENTITY / PASS
    # ---------------------------------------------------------
    reference.save(
        GENERATED_DIR / "generated_identical.jpeg",
        quality=95
    )

    # ---------------------------------------------------------
    # 2. Brightness change
    # Expected: SAME_IDENTITY / PASS
    # ---------------------------------------------------------
    brightness = ImageEnhance.Brightness(
        reference
    ).enhance(1.25)

    brightness.save(
        GENERATED_DIR / "generated_brightness.jpeg",
        quality=95
    )

    # ---------------------------------------------------------
    # 3. Moderate blur
    # Expected: identity may remain detectable,
    # but visual quality should degrade
    # ---------------------------------------------------------
    moderate_blur = reference.filter(
        ImageFilter.GaussianBlur(radius=5)
    )

    moderate_blur.save(
        GENERATED_DIR / "generated_moderate_blur.jpeg",
        quality=95
    )

    # ---------------------------------------------------------
    # 4. Heavy blur
    # Expected: strong visual degradation / FAIL
    # ---------------------------------------------------------
    heavy_blur = reference.filter(
        ImageFilter.GaussianBlur(radius=18)
    )

    heavy_blur.save(
        GENERATED_DIR / "generated_heavy_blur.jpeg",
        quality=95
    )

    # ---------------------------------------------------------
    # 5. Crop change
    # Expected: visual change / FAIL
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 6. Dedicated VISUAL DRIFT fixture
    #
    # This deliberately changes the overall visual appearance
    # while preserving the basic image dimensions.
    #
    # Operations:
    #   - horizontal flip
    #   - contrast reduction
    #   - grayscale conversion
    #   - colorization
    #
    # Expected:
    #   VISUAL_DRIFT
    #   FAIL
    #
    # This fixture is deterministic and reproducible.
    # ---------------------------------------------------------
    visual_drift = ImageOps.mirror(reference)

    visual_drift = ImageEnhance.Contrast(
        visual_drift
    ).enhance(0.55)

    visual_drift = ImageEnhance.Color(
        visual_drift
    ).enhance(0.20)

    visual_drift = ImageEnhance.Brightness(
        visual_drift
    ).enhance(0.85)

    visual_drift.save(
        GENERATED_DIR / "generated_visual_drift.jpeg",
        quality=95
    )

    # ---------------------------------------------------------
    # Output summary
    # ---------------------------------------------------------
    print("Fixture generation completed.")
    print(f"Output directory: {GENERATED_DIR}")
    print()

    for file in sorted(
        GENERATED_DIR.glob("generated_*.jpeg")
    ):
        print(f"  Created: {file.name}")


if __name__ == "__main__":
    generate_fixtures()