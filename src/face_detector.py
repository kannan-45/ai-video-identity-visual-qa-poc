import cv2


class FaceDetector:

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect(self, image_path):
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

        return faces


if __name__ == "__main__":
    detector = FaceDetector()

    faces = detector.detect(
        "mock_data/inputs/reference/reference_001.png"
    )

    print("Faces detected:", len(faces))