from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DINOV3_ROOT = (
    PROJECT_ROOT
    / "models"
    / "dinov3"
    / "dinov3-main"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "dinov3"
    / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)


# ============================================================
# DINOv3 checker
# ============================================================

class DINOv3Checker:
    """
    DINOv3-based visual similarity checker.

    Uses the official DINOv3 ViT-S/16 model and the locally
    downloaded official LVD-1689M checkpoint.

    Output:
        available
        score
        label
        reason_code
    """

    def __init__(self, threshold=0.80):

        self.threshold = threshold

        self.model = None
        self.device = torch.device("cpu")
        self.available = False
        self.error = None

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        self._load_model()

    # --------------------------------------------------------
    # Load official DINOv3 model
    # --------------------------------------------------------

    def _load_model(self):

        try:

            if not DINOV3_ROOT.exists():
                raise FileNotFoundError(
                    f"DINOv3 source directory not found: {DINOV3_ROOT}"
                )

            if not CHECKPOINT_PATH.exists():
                raise FileNotFoundError(
                    f"DINOv3 checkpoint not found: {CHECKPOINT_PATH}"
                )

            # Make the official DINOv3 package importable.
            dinov3_parent = str(DINOV3_ROOT)

            if dinov3_parent not in sys.path:
                sys.path.insert(0, dinov3_parent)

            # Import the official DINOv3 hub function.
            from dinov3.hub.backbones import dinov3_vits16

            print("Loading official DINOv3 ViT-S/16...")
            print(f"Checkpoint: {CHECKPOINT_PATH}")

            # The official loader accepts a local checkpoint path
            # through the 'weights' argument.
            self.model = dinov3_vits16(
                pretrained=True,
                weights=str(CHECKPOINT_PATH),
                check_hash=False,
            )

            self.model = self.model.to(self.device)
            self.model.eval()

            self.available = True

            print("DINOv3 loaded successfully.")
            print("Device: CPU")

        except Exception as exc:

            self.available = False
            self.model = None
            self.error = str(exc)

            print("DINOv3 could not be loaded.")
            print(f"Reason: {exc}")

    # --------------------------------------------------------
    # Image preprocessing
    # --------------------------------------------------------

    def _load_image(self, image_path):

        image = Image.open(image_path).convert("RGB")

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    def _extract_embedding(self, image_path):

        image_tensor = self._load_image(image_path)

        with torch.inference_mode():

            output = self.model(image_tensor)

            # DINOv3 hub models may expose CLS features through
            # different output formats depending on version.
            if isinstance(output, dict):

                if "x_norm_clstoken" in output:
                    embedding = output["x_norm_clstoken"]

                elif "x_norm_patchtokens" in output:
                    embedding = output["x_norm_patchtokens"].mean(dim=1)

                elif "x_prenorm" in output:
                    embedding = output["x_prenorm"][:, 0]

                elif "x" in output:
                    value = output["x"]

                    if value.ndim == 3:
                        embedding = value[:, 0]
                    else:
                        embedding = value

                else:
                    raise RuntimeError(
                        "Unable to find a usable feature tensor "
                        "in the DINOv3 output."
                    )

            elif isinstance(output, torch.Tensor):

                if output.ndim == 3:
                    embedding = output[:, 0]

                elif output.ndim == 2:
                    embedding = output

                else:
                    raise RuntimeError(
                        f"Unexpected DINOv3 tensor shape: {output.shape}"
                    )

            else:
                raise RuntimeError(
                    f"Unexpected DINOv3 output type: {type(output)}"
                )

            embedding = embedding.float()

            embedding = torch.nn.functional.normalize(
                embedding,
                dim=-1,
            )

        return embedding

    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    def _cosine_similarity(self, embedding_a, embedding_b):

        score = torch.sum(
            embedding_a * embedding_b,
            dim=-1,
        )

        return float(score.item())

    # --------------------------------------------------------
    # Public comparison method
    # --------------------------------------------------------

    def compare(self, reference_path, generated_path):

        if not self.available:

            return {
                "available": False,
                "label": "DINO3_UNAVAILABLE",
                "score": None,
                "reason_code": "DINOV3_LOAD_FAILED",
                "error": self.error,
            }

        try:

            reference_embedding = self._extract_embedding(
                reference_path
            )

            generated_embedding = self._extract_embedding(
                generated_path
            )

            score = self._cosine_similarity(
                reference_embedding,
                generated_embedding,
            )

            score = round(score, 4)

            if score >= self.threshold:

                label = "SIMILAR"
                reason_code = None

            else:

                label = "VISUAL_DRIFT"
                reason_code = "VISUAL_DINOV3_DRIFT"

            result = {
                "available": True,
                "model": "DINOv3 ViT-S/16",
                "score": score,
                "threshold": self.threshold,
                "label": label,
                "reason_code": reason_code,
            }

            return result

        except Exception as exc:

            return {
                "available": False,
                "label": "DINO3_UNAVAILABLE",
                "score": None,
                "reason_code": "DINOV3_INFERENCE_FAILED",
                "error": str(exc),
            }


# ============================================================
# Simple standalone test
# ============================================================

if __name__ == "__main__":

    reference = (
        PROJECT_ROOT
        / "mock_data"
        / "inputs"
        / "adaface"
        / "reference"
        / "reference_adaface.jpeg"
    )

    generated = (
        PROJECT_ROOT
        / "mock_data"
        / "inputs"
        / "adaface"
        / "generated"
        / "generated_visual_drift.jpeg"
    )

    checker = DINOv3Checker()

    result = checker.compare(
        str(reference),
        str(generated),
    )

    print("\nDINOv3 Result")
    print("=" * 60)

    for key, value in result.items():
        print(f"{key}: {value}")