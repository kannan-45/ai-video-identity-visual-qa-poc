import json
import os

from src.face_identity_checker import FaceIdentityChecker
from src.adaface_checker import AdaFaceChecker
from src.ssim_checker import calculate_ssim
from src.lpips_checker import LPIPSChecker
from src.clip_checker import CLIPChecker

class QAEngine:

    def __init__(self):
        self.face_checker = FaceIdentityChecker()
        self.adaface_checker = AdaFaceChecker()
        self.lpips_checker = LPIPSChecker()
        self.clip_checker = CLIPChecker()

    def analyze(self, reference_path, generated_path,output_path=None):

        # 1. Face identity
        identity_result = self.face_checker.compare(
            reference_path,
            generated_path
        )

        adaface_result = self.adaface_checker.compare(
            reference_path,
            generated_path
        )
        # 2. SSIM
        ssim_score = calculate_ssim(
            reference_path,
            generated_path
        )

        # 3. LPIPS
        lpips_score = self.lpips_checker.calculate(
            reference_path,
            generated_path
        )

        # 4. CLIP
        clip_score = self.clip_checker.calculate(
            reference_path,
            generated_path
        )

        # Reason codes
        reason_codes = []
        if adaface_result["label"] == "REFERENCE_FACE_MISSING":
            reason_codes.append("ID_REFERENCE_FACE_MISSING")
        if adaface_result["label"] == "GENERATED_FACE_MISSING":
            reason_codes.append("ID_FACE_MISSING")

        elif adaface_result["label"] == "DIFFERENT_IDENTITY":
            reason_codes.append("ID_FACE_MISMATCH")

        if ssim_score < 0.90:
            reason_codes.append("VISUAL_STRUCTURAL_CHANGE")

        if lpips_score > 0.30:
            reason_codes.append("VISUAL_PERCEPTUAL_CHANGE")

        if clip_score < 0.85:
            reason_codes.append("VISUAL_LOW_SIMILARITY")

        # Final decision
        if reason_codes:
            decision = "FAIL"
        else:
            decision = "PASS"

        result = {
            "identity": {
            "primary": {
                "model": "AdaFace",
                "score": adaface_result["score"],
                "label": adaface_result["label"]
            },
            "prototype_evidence": {
                "model": "PixelFacePrototype",
                "score": identity_result["score"],
                "label": identity_result["label"]
            
            
            }
        },
            "visual": {
                "ssim": ssim_score,
                "lpips": lpips_score,
                "clip": clip_score
            },
            "decision": decision,
            "reason_codes": reason_codes
        }
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

        return result

if __name__ == "__main__":

    engine = QAEngine()

    result = engine.analyze(
        "mock_data/inputs/reference/reference_001.png",
        "mock_data/inputs/generated/generated_001_same_identity.png"
    )

    output_path = "mock_data/expected/qa_report.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(result)
    print("QA report saved to:", output_path)