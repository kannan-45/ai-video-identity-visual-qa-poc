import sys
import os
import torch

# Add the official AdaFace repository to Python path
ADAFACE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Adaface-master", "AdaFace-master")
)

if ADAFACE_DIR not in sys.path:
    sys.path.insert(0, ADAFACE_DIR)

import inference
from face_alignment import align


class AdaFaceChecker:

    def __init__(self):
        self.model = inference.load_pretrained_model("ir_50")
        self.model.eval()

    def get_embedding(self, image_path):

        aligned_face = align.get_aligned_face(image_path)

        if aligned_face is None:
            return None

        input_tensor = inference.to_input(aligned_face)

        with torch.no_grad():
            feature, _ = self.model(input_tensor)

        return feature

    def compare(self, reference_path, generated_path):

        reference_embedding = self.get_embedding(reference_path)
        generated_embedding = self.get_embedding(generated_path)

        if reference_embedding is None:
            return {
                "label": "REFERENCE_FACE_MISSING",
                "score": 0.0
            }

        if generated_embedding is None:
            return {
                "label": "GENERATED_FACE_MISSING",
                "score": 0.0
            }

        similarity = torch.nn.functional.cosine_similarity(
            reference_embedding,
            generated_embedding
        ).item()

        similarity = round(float(similarity), 4)

        if similarity >= 0.40:
            label = "SAME_IDENTITY"
        else:
            label = "DIFFERENT_IDENTITY"

        return {
            "label": label,
            "score": similarity
        }


if __name__ == "__main__":

    checker = AdaFaceChecker()

    result = checker.compare(
        "Adaface-master/AdaFace-master/face_alignment/test_images/img1.jpeg",
        "Adaface-master/AdaFace-master/face_alignment/test_images/img3.jpeg"
    )

    print("AdaFace result:")
    print(result)