import torch
import open_clip
from PIL import Image


# Load CLIP model once
_device = "cpu"

_model, _, _preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai",
    device=_device
)

_model.eval()


def calculate_clip_similarity(reference_path, generated_path):
    """
    Calculate CLIP cosine similarity.

    Higher score = more semantically similar.
    Lower score = greater visual/semantic difference.
    """

    reference_image = _preprocess(
        Image.open(reference_path).convert("RGB")
    ).unsqueeze(0).to(_device)

    generated_image = _preprocess(
        Image.open(generated_path).convert("RGB")
    ).unsqueeze(0).to(_device)

    with torch.no_grad():
        reference_features = _model.encode_image(reference_image)
        generated_features = _model.encode_image(generated_image)

        reference_features = reference_features / reference_features.norm(
            dim=-1,
            keepdim=True
        )

        generated_features = generated_features / generated_features.norm(
            dim=-1,
            keepdim=True
        )

        similarity = (
            reference_features @ generated_features.T
        ).item()

    return round(float(similarity), 4)


if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

    score = calculate_clip_similarity(
        str(reference),
        str(generated)
    )

    print("CLIP similarity:", score)