from pathlib import Path

from src.face_identity_checker import FaceIdentityChecker
from src.adaface_checker import AdaFaceChecker
from src.ssim_checker import calculate_ssim
from src.lpips_checker import calculate_lpips
from src.clip_checker import calculate_clip_similarity
from src.dinov3_checker import DINOv3Checker
from src.object_environment_checker import ObjectEnvironmentChecker


class QAEngine:

    def __init__(self):
        self.face_checker = FaceIdentityChecker()
        self.adaface_checker = AdaFaceChecker()
        self.dinov3_checker = DINOv3Checker()
        self.object_environment_checker = ObjectEnvironmentChecker()

    def analyze(self, reference_path, generated_path):

        reason_codes = []

        # =====================================================
        # Identity
        # =====================================================

        reference_face = self.face_checker.detect_face(reference_path)
        generated_face = self.face_checker.detect_face(generated_path)

        if reference_face is None:
            reason_codes.append("ID_REFERENCE_FACE_MISSING")

        if generated_face is None:
            reason_codes.append("ID_FACE_MISSING")

        # AdaFace = primary identity model
        adaface_result = self.adaface_checker.compare(
            reference_path,
            generated_path,
        )

        if adaface_result["label"] == "DIFFERENT_IDENTITY":
            reason_codes.append("ID_FACE_MISMATCH")

        # Prototype model retained as supporting evidence
        prototype_result = self.face_checker.compare(
            reference_path,
            generated_path,
        )

        if (
            prototype_result.get("label") == "DIFFERENT_IDENTITY"
            and adaface_result.get("label") != "SAME_IDENTITY"
        ):
            reason_codes.append("ID_FACE_MISMATCH")

        # =====================================================
        # Visual metrics
        # =====================================================

        ssim_score = calculate_ssim(
            reference_path,
            generated_path,
        )

        lpips_score = calculate_lpips(
            reference_path,
            generated_path,
        )

        clip_score = calculate_clip_similarity(
            reference_path,
            generated_path,
        )

        # =====================================================
        # SSIM
        # =====================================================

        if ssim_score < 0.90:
            reason_codes.append("VISUAL_STRUCTURAL_CHANGE")

        # =====================================================
        # LPIPS
        # =====================================================

        if lpips_score > 0.30:
            reason_codes.append("VISUAL_PERCEPTUAL_CHANGE")

        # =====================================================
        # CLIP
        # =====================================================

        if clip_score < 0.85:
            reason_codes.append("VISUAL_LOW_SIMILARITY")

        # =====================================================
        # DINOv3
        # =====================================================

        dinov3_result = self.dinov3_checker.compare(
            reference_path,
            generated_path,
        )

        if (
            dinov3_result.get("available")
            and dinov3_result.get("label") == "VISUAL_DRIFT"
        ):
            reason_codes.append("VISUAL_DINOV3_DRIFT")

        # =====================================================
        # Object + environment consistency
        # =====================================================

        object_environment_result = (
            self.object_environment_checker.compare(
                reference_path,
                generated_path,
            )
        )

        object_result = object_environment_result.get(
            "object",
            {},
        )

        environment_result = object_environment_result.get(
            "environment",
            {},
        )

        if object_result.get("label") == "OBJECT_DRIFT":
            reason_codes.append("VISUAL_OBJECT_DRIFT")

        if environment_result.get("label") == "ENVIRONMENT_DRIFT":
            reason_codes.append("VISUAL_ENVIRONMENT_DRIFT")

        # =====================================================
        # Multi-signal fallback
        #
        # Only active when DINOv3 is unavailable.
        # =====================================================

        dinov3_available = bool(
            dinov3_result.get("available")
        )

        fallback_triggered = False

        if not dinov3_available:

            if (
                ssim_score < 0.75
                and lpips_score > 0.15
                and clip_score < 0.95
            ):
                fallback_triggered = True

                reason_codes.append(
                    "VISUAL_DRIFT_FALLBACK"
                )

        fallback_reason = (
            "DINOv3 available; fallback not required."
            if dinov3_available
            else
            "DINOv3 unavailable; fallback uses "
            "SSIM + LPIPS + CLIP to detect strong "
            "multi-signal visual change."
        )

        # =====================================================
        # Final decision
        # =====================================================

        decision = (
            "FAIL"
            if reason_codes
            else
            "PASS"
        )

        # Remove duplicate reason codes while preserving order
        reason_codes = list(dict.fromkeys(reason_codes))

        # =====================================================
        # Final report
        # =====================================================

        report = {
            "identity": {
                "primary": {
                    "model": "AdaFace",
                    "score": adaface_result.get("score"),
                    "label": adaface_result.get("label"),
                },
                "prototype_evidence": {
                    "model": "PixelFacePrototype",
                    "score": prototype_result.get("score"),
                    "label": prototype_result.get("label"),
                },
            },

            "visual": {
                "ssim": ssim_score,
                "lpips": lpips_score,
                "clip": clip_score,

                "dinov3": dinov3_result,

                "object_environment": (
                    object_environment_result
                ),

                "visual_drift_fallback": {
                    "enabled": not dinov3_available,
                    "triggered": fallback_triggered,
                    "method": "SSIM + LPIPS + CLIP",
                    "reason": fallback_reason,
                },
            },

            "decision": decision,

            "reason_codes": reason_codes,
        }

        return report


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    reference = (
        project_root
        / "mock_data"
        / "inputs"
        / "adaface"
        / "reference"
        / "reference_adaface.jpeg"
    )

    generated = (
        project_root
        / "mock_data"
        / "inputs"
        / "adaface"
        / "generated"
        / "generated_visual_drift.jpeg"
    )

    engine = QAEngine()

    result = engine.analyze(
        str(reference),
        str(generated),
    )

    import json

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    output_path = (
        project_root
        / "mock_data"
        / "expected"
        / "visual_drift_qa_report.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print(
        f"\nQA report saved to: {output_path}"
    )