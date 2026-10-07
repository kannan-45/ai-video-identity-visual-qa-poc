from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
REFERENCE = (
    ROOT
    / "inputs"
    / "adaface"
    / "reference"
    / "reference_adaface.jpeg"
)

OUTPUT = (
    ROOT
    / "inputs"
    / "member7"
    / "pan_zoom_frames"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


def main():
    image = Image.open(
        REFERENCE
    ).convert("RGB")

    width, height = image.size

    print(
        f"Reference size: "
        f"{width}x{height}"
    )

    for frame_index in range(25):

        # Increasing crop amount creates
        # deterministic zoom across the clip.
        progress = frame_index / 24.0

        margin_x = int(
            width * 0.02 * progress
        )

        margin_y = int(
            height * 0.02 * progress
        )

        left = margin_x
        top = margin_y
        right = width - margin_x
        bottom = height - margin_y

        cropped = image.crop(
            (
                left,
                top,
                right,
                bottom,
            )
        )

        frame = cropped.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )

        output_path = (
            OUTPUT
            / f"frame_{frame_index:03d}.png"
        )

        frame.save(
            output_path
        )

        print(
            f"Created: {output_path}"
        )


if __name__ == "__main__":
    main()