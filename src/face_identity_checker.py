import cv2
import numpy as np


class FaceIdentityChecker:

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect_face(self, image_path):
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return None

        face = max(faces, key=lambda box: box[2] * box[3])

        x, y, w, h = face

        return image[y:y+h, x:x+w]

    def compare(self, reference_path, generated_path):

        reference_face = self.detect_face(reference_path)
        generated_face = self.detect_face(generated_path)

        if reference_face is None:
            return {
                "label": "REFERENCE_FACE_MISSING",
                "score": 0.0
            }

        if generated_face is None:
            return {
                "label": "GENERATED_FACE_MISSING",
                "score": 0.0
            }

        reference_face = cv2.resize(reference_face, (128, 128))
        generated_face = cv2.resize(generated_face, (128, 128))

        reference_gray = cv2.cvtColor(reference_face, cv2.COLOR_BGR2GRAY)
        generated_gray = cv2.cvtColor(generated_face, cv2.COLOR_BGR2GRAY)

        difference = np.mean(
            np.abs(
                reference_gray.astype(np.float32)
                - generated_gray.astype(np.float32)
            )
        )

        score = max(0.0, 1.0 - (difference / 255.0))

        score = round(float(score), 4)

        if score >= 0.90:
            label = "SAME_IDENTITY"
        else:
            label = "DIFFERENT_IDENTITY"

        return {
            "label": label,
            "score": score
        }


if __name__ == "__main__":

    checker = FaceIdentityChecker()

    result = checker.compare(
        "mock_data/inputs/reference/reference_001.png",
        "mock_data/inputs/generated/generated_001_same_identity.png"
    )

    print("Face identity result:")
    print(result)