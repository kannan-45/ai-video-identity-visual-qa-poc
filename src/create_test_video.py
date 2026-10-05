import cv2
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "mock_data" / "inputs"


def create_test_video(output_path, add_failure=False):
    """
    Create a deterministic mock video for Keyframe QA.

    add_failure=False:
        All frames are normal -> expected PASS.

    add_failure=True:
        Frames 12-14 are deliberately blurred -> expected FAIL.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 640
    height = 480
    fps = 5
    total_frames = 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open video writer: {output_path}"
        )

    # Load reference image used by the QA pipeline.
    reference_path = (
        PROJECT_ROOT
        / "mock_data"
        / "inputs"
        / "adaface"
        / "reference"
        / "reference_adaface.jpeg"
    )

    reference = cv2.imread(str(reference_path))

    if reference is None:
        raise FileNotFoundError(
            f"Reference image not found: {reference_path}"
        )

    reference = cv2.resize(
        reference,
        (width, height)
    )

    for frame_number in range(total_frames):

        frame = reference.copy()

        # Deliberate failure only when requested.
        if add_failure and 12 <= frame_number <= 14:

            frame = cv2.GaussianBlur(
                frame,
                (51, 51),
                0
            )

        writer.write(frame)

    writer.release()

    print(f"Test video created:")
    print(output_path)

    if add_failure:
        print()
        print("Video contents:")
        print("Frames 0-11 : Normal")
        print("Frames 12-14: DELIBERATE FAILURE")
        print("Frames 15-24: Normal")
    else:
        print()
        print("Video contents:")
        print("Frames 0-24: Normal")
        print("Expected result: PASS")


if __name__ == "__main__":

    # Deliberate-failure video
    failure_video = OUTPUT_DIR / "test_video.mp4"

    create_test_video(
        failure_video,
        add_failure=True
    )

    # Clean video
    clean_video = OUTPUT_DIR / "clean_test_video.mp4"

    create_test_video(
        clean_video,
        add_failure=False
    )