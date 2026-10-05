import json
import os

from src.object_environment_checker import ObjectEnvironmentChecker
from src.face_identity_checker import FaceIdentityChecker
from src.adaface_checker import AdaFaceChecker
from src.ssim_checker import calculate_ssim
from src.lpips_checker import LPIPSChecker
from src.clip_checker import CLIPChecker
from src.dinov3_checker import DINOv3Checker


class QAEngine:

    def __init__(self):
        self.face_checker = FaceIdentityChecker()
        self.adaface_checker = AdaFaceChecker()
        self.lpips_checker = LPIPSChecker()
        self.clip_checker = CLIPChecker()
        self.dinov3_checker = DINOv3Checker()
        self.object_environment_checker = ObjectEnvironmentChecker()

    def analyze(
        self,
        reference_path,
        generated_path,
        output_path=None
    ):

        # --------------------------------------------------
        # 1. Face identity - prototype evidence
        # --------------------------------------------------

        identity_result = self.face_checker.compare(
            reference_path,
            generated_path
        )

        # --------------------------------------------------
        # 2. AdaFace identity
        # --------------------------------------------------

        adaface_result = self.adaface_checker.compare(
            reference_path,
            generated_path
        )

        # --------------------------------------------------
        # 3. SSIM - structural similarity
        # --------------------------------------------------

        ssim_score = calculate_ssim(
            reference_path,
            generated_path
        )

        # --------------------------------------------------
        # 4. LPIPS - perceptual similarity
        # --------------------------------------------------

        lpips_score = self.lpips_checker.calculate(
            reference_path,
            generated_path
        )

        # --------------------------------------------------
        # 5. CLIP - semantic similarity
        # --------------------------------------------------

        clip_score = self.clip_checker.calculate(
            reference_path,
            generated_path
        )

        # --------------------------------------------------
        # 6. DINOv3 - visual feature similarity
        # --------------------------------------------------

        dinov3_result = self.dinov3_checker.compare(
            reference_path,
            generated_path
        )

        # --------------------------------------------------
        # 7. Object and environment consistency
        # --------------------------------------------------

        object_environment_result = (
            self.object_environment_checker.compare(
                reference_path,
                generated_path
            )
        )

        # --------------------------------------------------
        # Reason codes
        # --------------------------------------------------

        reason_codes = []

        # --------------------------------------------------
        # Identity checks
        # --------------------------------------------------

        if adaface_result["label"] == "REFERENCE_FACE_MISSING":

            reason_codes.append(
                "ID_REFERENCE_FACE_MISSING"
            )

        elif adaface_result["label"] == "GENERATED_FACE_MISSING":

            reason_codes.append(
                "ID_FACE_MISSING"
            )

        elif adaface_result["label"] == "DIFFERENT_IDENTITY":

            reason_codes.append(
                "ID_FACE_MISMATCH"
            )

        # --------------------------------------------------
        # Structural similarity
        # --------------------------------------------------

        if ssim_score < 0.90:

            reason_codes.append(
                "VISUAL_STRUCTURAL_CHANGE"
            )

        # --------------------------------------------------
        # Perceptual similarity
        # --------------------------------------------------

        if lpips_score > 0.30:

            reason_codes.append(
                "VISUAL_PERCEPTUAL_CHANGE"
            )

        # --------------------------------------------------
        # Semantic similarity
        # --------------------------------------------------

        if clip_score < 0.85:

            reason_codes.append(
                "VISUAL_LOW_SIMILARITY"
            )

        # --------------------------------------------------
        # DINOv3
        #
        # If DINOv3 is available, it is the primary
        # visual-drift detector.
        #
        # If DINOv3 is unavailable, the system falls back
        # to the structural/perceptual evidence above.
        # --------------------------------------------------

        if dinov3_result["available"]:

            if dinov3_result["label"] == "VISUAL_DRIFT":

                reason_codes.append(
                    "VISUAL_DINOV3_DRIFT"
                )

        # --------------------------------------------------
        # Object consistency
        # --------------------------------------------------

        if (
            object_environment_result["object"]["label"]
            == "OBJECT_DRIFT"
        ):

            reason_codes.append(
                "VISUAL_OBJECT_DRIFT"
            )

        # --------------------------------------------------
        # Environment consistency
        # --------------------------------------------------

        if (
            object_environment_result["environment"]["label"]
            == "ENVIRONMENT_DRIFT"
        ):

            reason_codes.append(
                "VISUAL_ENVIRONMENT_DRIFT"
            )

        # --------------------------------------------------
        # Explicit visual-drift fallback
        #
        # DINOv3 may be unavailable because its official
        # checkpoint requires gated access.
        #
        # We therefore classify deliberate strong visual
        # changes using multiple independent signals.
        #
        # This does NOT pretend to be DINOv3.
        # It is explicitly labelled as fallback evidence.
        # --------------------------------------------------

        visual_drift_fallback = False

        if (
            ssim_score < 0.75
            and lpips_score > 0.15
            and clip_score < 0.95
        ):

            visual_drift_fallback = True

            reason_codes.append(
                "VISUAL_DRIFT_FALLBACK"
            )

        # --------------------------------------------------
        # Final decision
        # --------------------------------------------------

        if reason_codes:

            decision = "FAIL"

        else:

            decision = "PASS"

        # --------------------------------------------------
        # QA report
        # --------------------------------------------------

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

                "clip": clip_score,

                "dinov3": dinov3_result,

                "object_environment":
                    object_environment_result,

                "visual_drift_fallback": {

                    "enabled": not dinov3_result["available"],

                    "triggered": visual_drift_fallback,

                    "method":
                        "SSIM + LPIPS + CLIP",

                    "reason":
                        (
                            "DINOv3 checkpoint unavailable; "
                            "fallback detects strong multi-signal "
                            "visual change."
                        )
                }
            },

            "decision": decision,

            "reason_codes": reason_codes
        }

        # --------------------------------------------------
        # Save report
        # --------------------------------------------------

        if output_path:

            output_directory = os.path.dirname(
                output_path
            )

            if output_directory:

                os.makedirs(
                    output_directory,
                    exist_ok=True
                )

            with open(
                output_path,
                "w"
            ) as f:

                json.dump(
                    result,
                    f,
                    indent=2
                )

        return result


if __name__ == "__main__":

    engine = QAEngine()

    reference_path = (
        "mock_data/inputs/adaface/reference/"
        "reference_adaface.jpeg"
    )

    generated_path = (
        "mock_data/inputs/adaface/generated/"
        "generated_visual_drift.jpeg"
    )

    output_path = (
        "mock_data/expected/"
        "visual_drift_qa_report.json"
    )

    result = engine.analyze(
        reference_path,
        generated_path,
        output_path
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    print()
    print(
        "QA report saved to:",
        output_path
    )