import cv2
from pathlib import Path

REFERENCE = Path(
    "mock_data/inputs/adaface/reference/reference_adaface.jpeg"
)

OUTPUT = Path(
    "mock_data/inputs/test_video.mp4"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(REFERENCE))

if image is None:
    raise FileNotFoundError(
        f"Reference image not found: {REFERENCE}"
    )

height, width = image.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video = cv2.VideoWriter(
    str(OUTPUT),
    fourcc,
    5.0,
    (width, height)
)

# Create 25 frames.
# Most frames are identical to the reference.
# Frames 12-14 contain a deliberate visual defect.

for frame_number in range(25):

    frame = image.copy()

    # Deliberate failure:
    # Replace the face region with a heavily blurred version.
    if 12 <= frame_number <= 14:

        # Blur the entire frame strongly.
        frame = cv2.GaussianBlur(
            frame,
            (51, 51),
            0
        )

    video.write(frame)

video.release()

print("Test video created:")
print(OUTPUT)

print("\nVideo contents:")
print("Frames 0-11 : Normal")
print("Frames 12-14: DELIBERATE FAILURE")
print("Frames 15-24: Normal")