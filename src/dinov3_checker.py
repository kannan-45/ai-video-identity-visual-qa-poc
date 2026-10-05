import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DINOv3Checker:
    """
    DINOv3 visual similarity checker.

    The model is optional because the official DINOv3
    pretrained weights require access to the gated checkpoint.

    Once access is available, set DINOV3_MODEL to the
    approved DINOv3 model/checkpoint.
    """

    def __init__(
        self,
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
    ):
        self.model_name = model_name
        self.processor = None
        self.model = None

    def load_model(self):
        """
        Load the DINOv3 processor and model.

        This is intentionally separate from __init__ so that
        the rest of the QA pipeline can run without DINOv3
        when the gated checkpoint is unavailable.
        """

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name
        )

        self.model = AutoModel.from_pretrained(
            self.model_name
        )

        self.model.eval()

    def get_embedding(self, image_path):
        """
        Generate a normalized DINOv3 image embedding.
        """

        if self.model is None:
            self.load_model()

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # CLS token representation
        embedding = outputs.last_hidden_state[:, 0, :]

        embedding = torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding

    def compare(self, reference_path, generated_path):
        """
        Compare two images using cosine similarity.
        """

        try:
            reference_embedding = self.get_embedding(
                reference_path
            )

            generated_embedding = self.get_embedding(
                generated_path
            )

        except Exception as exc:
            return {
                "available": False,
                "label": "DINO3_UNAVAILABLE",
                "score": None,
                "reason_code": "DINOV3_CHECKPOINT_ACCESS_REQUIRED"
            }

        similarity = torch.nn.functional.cosine_similarity(
            reference_embedding,
            generated_embedding
        ).item()

        similarity = round(float(similarity), 4)

        return {
            "available": True,
            "label": "SIMILAR" if similarity >= 0.80 else "VISUAL_DRIFT",
            "score": similarity
        }