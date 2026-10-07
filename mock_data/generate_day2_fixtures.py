from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "inputs" / "adaface" / "reference" / "reference_adaface.jpeg"
OUTPUT = ROOT / "inputs" / "adaface" / "generated"

OUTPUT.mkdir(parents=True, exist_ok=True)


def save(image, filename):
    path = OUTPUT / filename
    image.save(path, quality=95)
    print(f"Created: {path}")


def main():
    original = Image.open(REFERENCE).convert("RGB")

    # 14 — Brightness increase
    bright = ImageEnhance.Brightness(original).enhance(1.35)
    save(bright, "generated_brightness_high.jpeg")

    # 15 — Brightness decrease
    dark = ImageEnhance.Brightness(original).enhance(0.65)
    save(dark, "generated_brightness_low.jpeg")

    # 16 — Colour shift
    colour = ImageEnhance.Color(original).enhance(1.8)
    save(colour, "generated_colour_shift.jpeg")

    # 17 — Strong crop
    width, height = original.size
    margin_x = int(width * 0.18)
    margin_y = int(height * 0.18)

    cropped = original.crop(
        (
            margin_x,
            margin_y,
            width - margin_x,
            height - margin_y,
        )
    )

    cropped = cropped.resize(original.size)
    save(cropped, "generated_strong_crop.jpeg")

    # 18 — Mild blur
    mild_blur = original.filter(ImageFilter.GaussianBlur(radius=2))
    save(mild_blur, "generated_mild_blur.jpeg")

    # 19 — Strong visual drift
    drift = ImageEnhance.Contrast(original).enhance(0.35)
    drift = ImageEnhance.Brightness(drift).enhance(1.25)
    drift = drift.filter(ImageFilter.GaussianBlur(radius=3))
    save(drift, "generated_strong_visual_drift.jpeg")

    # 20 — Pan/zoom-style transformation
    zoom_margin_x = int(width * 0.10)
    zoom_margin_y = int(height * 0.10)

    zoomed = original.crop(
        (
            zoom_margin_x,
            zoom_margin_y,
            width - zoom_margin_x,
            height - zoom_margin_y,
        )
    )

    zoomed = zoomed.resize(original.size)
    save(zoomed, "generated_zoom_transform.jpeg")


if __name__ == "__main__":
    main()