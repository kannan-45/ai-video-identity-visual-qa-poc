from pathlib import Path
import json

import cv2
import numpy as np

from .face_identity_checker import FaceIdentityChecker
from .adaface_checker import AdaFaceChecker
from .ssim_checker import calculate_ssim
from .lpips_checker import calculate_lpips
from .clip_checker import calculate_clip_similarity
from .dinov3_checker import DINOv3Checker
from .object_environment_checker import ObjectEnvironmentChecker


class QAEngine:
    """
    Identity + visual consistency QA engine.

    Day 1:
        Reference image -> generated image -> component scores -> QA report

    Day 2:
        Reference image + MP4 clip
        -> deterministic frame sampling
        -> frame-level QA
        -> clip-level aggregation
        -> shared QAResult
    """

    def __init__(self):
        self.face_checker = FaceIdentityChecker()
        self.adaface_checker = AdaFaceChecker()
        self.dinov3_checker = DINOv3Checker()
        self.object_environment_checker = ObjectEnvironmentChecker()

    # ================================================================
    # DAY 1 - IMAGE QA
    # ================================================================

    def analyze(self, reference_path, generated_path):
        """
        Analyze one reference image against one generated image.
        """

        reference_path = Path(reference_path)
        generated_path = Path(generated_path)

        reason_codes = []

        # ------------------------------------------------------------
        # Identity - prototype checker
        # ------------------------------------------------------------

        identity_report = self.face_checker.compare(
            str(reference_path),
            str(generated_path)
        )

        identity_label = identity_report.get("label")
        identity_score = identity_report.get("score")

        if identity_label == "REFERENCE_FACE_MISSING":
            reason_codes.append(
                "ID_REFERENCE_FACE_MISSING"
            )

        elif identity_label == "GENERATED_FACE_MISSING":
            reason_codes.append(
                "ID_FACE_MISSING"
            )

        elif identity_label == "DIFFERENT_IDENTITY":
            reason_codes.append(
                "ID_FACE_MISMATCH"
            )

        # ------------------------------------------------------------
        # Identity - AdaFace
        # ------------------------------------------------------------

        adaface_report = self.adaface_checker.compare(
            str(reference_path),
            str(generated_path)
        )

        adaface_label = adaface_report.get("label")
        adaface_score = adaface_report.get("score")

        if adaface_label == "DIFFERENT_IDENTITY":
            if "ID_FACE_MISMATCH" not in reason_codes:
                reason_codes.append(
                    "ID_FACE_MISMATCH"
                )

        # ------------------------------------------------------------
        # SSIM
        # ------------------------------------------------------------

        ssim_score = calculate_ssim(
            str(reference_path),
            str(generated_path)
        )

        if ssim_score < 0.90:
            reason_codes.append(
                "VISUAL_STRUCTURAL_CHANGE"
            )

        # ------------------------------------------------------------
        # LPIPS
        # ------------------------------------------------------------

        lpips_score = calculate_lpips(
            str(reference_path),
            str(generated_path)
        )

        if lpips_score > 0.30:
            reason_codes.append(
                "VISUAL_PERCEPTUAL_CHANGE"
            )

        # ------------------------------------------------------------
        # CLIP
        # ------------------------------------------------------------

        clip_score = calculate_clip_similarity(
            str(reference_path),
            str(generated_path)
        )

        if clip_score < 0.85:
            reason_codes.append(
                "VISUAL_LOW_SIMILARITY"
            )

        # ------------------------------------------------------------
        # DINOv3
        # ------------------------------------------------------------

        dinov3_report = self.dinov3_checker.compare(
            str(reference_path),
            str(generated_path)
        )

        if dinov3_report.get("available", False):

            if dinov3_report.get("label") == "VISUAL_DRIFT":

                reason_code = dinov3_report.get(
                    "reason_code",
                    "VISUAL_DINOV3_DRIFT"
                )

                if reason_code not in reason_codes:
                    reason_codes.append(
                        reason_code
                    )

        # ------------------------------------------------------------
        # Object + environment
        # ------------------------------------------------------------

        object_environment_report = (
            self.object_environment_checker.compare(
                str(reference_path),
                str(generated_path)
            )
        )

        object_report = object_environment_report.get(
            "object",
            {}
        )

        environment_report = object_environment_report.get(
            "environment",
            {}
        )

        if object_report.get("label") == "OBJECT_DRIFT":
            reason_codes.append(
                "VISUAL_OBJECT_DRIFT"
            )

        if environment_report.get("label") == "ENVIRONMENT_DRIFT":
            reason_codes.append(
                "VISUAL_ENVIRONMENT_DRIFT"
            )

        # ------------------------------------------------------------
        # Fallback visual drift
        # ------------------------------------------------------------

        fallback_triggered = False

        if not dinov3_report.get("available", False):

            if (
                ssim_score < 0.75
                and lpips_score > 0.15
                and clip_score < 0.95
            ):
                fallback_triggered = True

                reason_codes.append(
                    "VISUAL_DRIFT_FALLBACK"
                )

        # ------------------------------------------------------------
        # Remove duplicate reason codes
        # ------------------------------------------------------------

        reason_codes = list(
            dict.fromkeys(reason_codes)
        )

        # ------------------------------------------------------------
        # Day 1 decision
        # ------------------------------------------------------------

        if reason_codes:
            decision = "FAIL"
        else:
            decision = "PASS"

        # ------------------------------------------------------------
        # Report
        # ------------------------------------------------------------

        report = {
            "reference": str(reference_path),

            "generated": str(generated_path),

            "identity": {
                "primary": {
                    "model": "AdaFace",
                    "score": adaface_score,
                    "label": adaface_label
                },

                "prototype_evidence": {
                    "model": "PixelFacePrototype",
                    "score": identity_score,
                    "label": identity_label
                }
            },

            "visual": {
                "ssim": ssim_score,

                "lpips": lpips_score,

                "clip": clip_score,

                "dinov3": dinov3_report,

                "object_environment": {
                    "object": object_report,
                    "environment": environment_report
                },

                "visual_drift_fallback": {
                    "enabled": not dinov3_report.get(
                        "available",
                        False
                    ),

                    "triggered": fallback_triggered,

                    "method": "SSIM + LPIPS + CLIP",

                    "reason": (
                        "DINOv3 available; "
                        "fallback not required."
                        if dinov3_report.get(
                            "available",
                            False
                        )
                        else
                        "DINOv3 unavailable; "
                        "fallback evaluated."
                    )
                }
            },

            "decision": decision,

            "reason_codes": reason_codes
        }

        return report

    # ================================================================
    # DAY 2 - DETERMINISTIC VIDEO FRAME SAMPLING
    # ================================================================

    def sample_video_frames(
        self,
        video_path,
        output_dir,
        num_frames=5
    ):
        """
        Deterministically sample evenly spaced frames.

        Example:

        25 frames + 5 samples

        -> 0
        -> 6
        -> 12
        -> 18
        -> 24
        """

        video_path = Path(video_path)
        output_dir = Path(output_dir)

        frames_dir = (
            output_dir / "frames"
        )

        frames_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Unable to open video: {video_path}"
            )

        total_frames = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
        )

        if total_frames <= 0:
            capture.release()

            raise RuntimeError(
                f"Video contains no frames: "
                f"{video_path}"
            )

        if fps <= 0:
            fps = 1.0

        sample_count = min(
            num_frames,
            total_frames
        )

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            sample_count,
            dtype=int
        )

        frame_indices = list(
            dict.fromkeys(
                frame_indices.tolist()
            )
        )

        sampled_frames = []

        for sample_number, frame_index in enumerate(
            frame_indices,
            start=1
        ):

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index
            )

            success, frame = capture.read()

            if not success:
                capture.release()

                raise RuntimeError(
                    f"Unable to read frame "
                    f"{frame_index} from "
                    f"{video_path}"
                )

            timestamp_s = (
                frame_index / fps
            )

            output_file = (
                frames_dir
                / (
                    f"frame_{sample_number:02d}"
                    f"_index_{frame_index:06d}.jpg"
                )
            )

            written = cv2.imwrite(
                str(output_file),
                frame
            )

            if not written:
                capture.release()

                raise RuntimeError(
                    f"Unable to write frame: "
                    f"{output_file}"
                )

            sampled_frames.append(
                {
                    "sample_number": sample_number,

                    "frame_index": int(
                        frame_index
                    ),

                    "timestamp_s": round(
                        timestamp_s,
                        4
                    ),

                    "file": str(
                        output_file
                    )
                }
            )

        capture.release()

        return {
            "video": str(video_path),

            "total_frames": total_frames,

            "fps": fps,

            "requested_frames": num_frames,

            "sampling_rule": (
                "Evenly spaced frame positions "
                "from first frame through last frame."
            ),

            "sampled_frames": sampled_frames
        }

    # ================================================================
    # DAY 2 - CLIP DECISION
    # ================================================================

    def _determine_clip_decision(
        self,
        frame_reports
    ):
        """
        Convert frame-level results into the shared
        Day 2 decision vocabulary.

        PASS:
            No sampled frame contains a failure.

        AUTO_RETRY:
            A clear identity or visual drift is detected.

        HUMAN_REVIEW:
            Visual degradation exists but does not
            meet the automatic retry criteria.
        """

        clear_failure_codes = {
            "ID_FACE_MISMATCH",
            "ID_FACE_MISSING",

            "VISUAL_DINOV3_DRIFT",

            "VISUAL_OBJECT_DRIFT",

            "VISUAL_ENVIRONMENT_DRIFT",

            "VISUAL_DRIFT_FALLBACK"
        }

        review_codes = {
            "VISUAL_STRUCTURAL_CHANGE",

            "VISUAL_PERCEPTUAL_CHANGE",

            "VISUAL_LOW_SIMILARITY"
        }

        all_reason_codes = []

        for report in frame_reports:

            all_reason_codes.extend(
                report.get(
                    "reason_codes",
                    []
                )
            )

        # Remove duplicates while preserving order.
        all_reason_codes = list(
            dict.fromkeys(
                all_reason_codes
            )
        )

        # ------------------------------------------------------------
        # AUTO_RETRY has highest priority
        # ------------------------------------------------------------

        if any(
            code in clear_failure_codes
            for code in all_reason_codes
        ):
            return "AUTO_RETRY"

        # ------------------------------------------------------------
        # HUMAN_REVIEW
        # ------------------------------------------------------------

        if any(
            code in review_codes
            for code in all_reason_codes
        ):
            return "HUMAN_REVIEW"

        # ------------------------------------------------------------
        # PASS
        # ------------------------------------------------------------

        return "PASS"

    # ================================================================
    # DAY 2 - VIDEO QA
    # ================================================================

    def analyze_clip(
        self,
        reference_path,
        video_path,
        output_dir=None,
        num_frames=5,
        qa_id="qa_clip_001",
        clip_id="clip_001"
    ):
        """
        Compare a reference image against frames
        sampled from an MP4 video.

        Returns a shared QAResult-style object.
        """

        reference_path = Path(
            reference_path
        )

        video_path = Path(
            video_path
        )

        # ------------------------------------------------------------
        # Output directory
        # ------------------------------------------------------------

        if output_dir is None:

            output_dir = (
                Path("mock_data")
                / "outputs"
                / clip_id
            )

        else:

            output_dir = Path(
                output_dir
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ------------------------------------------------------------
        # Sample frames
        # ------------------------------------------------------------

        sampling = self.sample_video_frames(
            video_path=video_path,

            output_dir=output_dir,

            num_frames=num_frames
        )

        frame_reports = []

        evidence_files = []

        # ------------------------------------------------------------
        # Analyze each sampled frame
        # ------------------------------------------------------------

        for sampled_frame in sampling[
            "sampled_frames"
        ]:

            frame_path = Path(
                sampled_frame["file"]
            )

            frame_report = self.analyze(
                reference_path,
                frame_path
            )

            frame_result = {
                "sample_number": sampled_frame[
                    "sample_number"
                ],

                "frame_index": sampled_frame[
                    "frame_index"
                ],

                "timestamp_s": sampled_frame[
                    "timestamp_s"
                ],

                "identity": frame_report[
                    "identity"
                ],

                "visual": frame_report[
                    "visual"
                ],

                "decision": frame_report[
                    "decision"
                ],

                "reason_codes": frame_report[
                    "reason_codes"
                ]
            }

            frame_reports.append(
                frame_result
            )

            evidence_files.append(
                str(frame_path)
            )

        # ------------------------------------------------------------
        # Clip-level decision
        # ------------------------------------------------------------

        clip_decision = (
            self._determine_clip_decision(
                frame_reports
            )
        )

        # ------------------------------------------------------------
        # Aggregate reason codes
        # ------------------------------------------------------------

        reason_codes = []

        for frame_report in frame_reports:

            for code in frame_report.get(
                "reason_codes",
                []
            ):

                if code not in reason_codes:

                    reason_codes.append(
                        code
                    )

        # ------------------------------------------------------------
        # Shared QAResult
        # ------------------------------------------------------------

        qa_result = {
            "qa_id": qa_id,

            "clip_id": clip_id,

            "qa_type": "IDENTITY_VISUAL",

            "component_scores": {
                "frames": frame_reports
            },

            "reason_codes": reason_codes,

            "decision": clip_decision,

            "evidence_files": evidence_files,

            "sampling": sampling
        }

        # ------------------------------------------------------------
        # Save QAResult
        # ------------------------------------------------------------

        qa_result_path = (
            output_dir
            / f"{clip_id}_QAResult.json"
        )

        with open(
            qa_result_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                qa_result,
                file,
                indent=2
            )

        # Add QAResult file to returned evidence.
        qa_result[
            "evidence_files"
        ].append(
            str(qa_result_path)
        )

        return qa_result