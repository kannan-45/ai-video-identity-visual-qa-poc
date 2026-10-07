from pathlib import Path
import hashlib
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
    Member 7 - Identity & Visual Consistency QA Engine.

    Day 1:
        Reference image vs generated image.

    Day 2:
        Reference image vs sampled video frames.
        ClipMetadata consumption and verification.
        Deterministic frame sampling.
        QAResult generation.
    """

    # ================================================================
    # Constructor
    # ================================================================

    def __init__(self):
        self.face_checker = FaceIdentityChecker()
        self.adaface_checker = AdaFaceChecker()
        self.dinov3_checker = DINOv3Checker()
        self.object_environment_checker = ObjectEnvironmentChecker()

    # ================================================================
    # Utility
    # ================================================================

    @staticmethod
    def _add_reason(reason_codes, reason):
        if reason not in reason_codes:
            reason_codes.append(reason)

    # ================================================================
    # Identity checks
    # ================================================================

    def _run_identity_checks(
        self,
        reference_path,
        generated_path,
    ):
        """
        Run primary AdaFace identity checking and the
        original Day 1 prototype face checker.
        """

        adaface_result = self.adaface_checker.compare(
            reference_path,
            generated_path,
        )

        prototype_result = self.face_checker.compare(
            reference_path,
            generated_path,
        )

        return {
            "primary": {
                "model": "AdaFace",
                "score": float(
                    adaface_result["score"]
                ),
                "label": adaface_result["label"],
            },
            "prototype_evidence": {
                "model": "PixelFacePrototype",
                "score": float(
                    prototype_result["score"]
                ),
                "label": prototype_result["label"],
            },
        }

    # ================================================================
    # Visual checks
    # ================================================================

    def _run_visual_checks(
        self,
        reference_path,
        generated_path,
    ):
        """
        Run visual consistency checks.
        """

        # ------------------------------------------------------------
        # SSIM
        # ------------------------------------------------------------

        ssim_score = float(
            calculate_ssim(
                reference_path,
                generated_path,
            )
        )

        # ------------------------------------------------------------
        # LPIPS
        # ------------------------------------------------------------

        lpips_score = float(
            calculate_lpips(
                reference_path,
                generated_path,
            )
        )

        # ------------------------------------------------------------
        # CLIP
        # ------------------------------------------------------------

        clip_score = float(
            calculate_clip_similarity(
                reference_path,
                generated_path,
            )
        )

        # ------------------------------------------------------------
        # DINOv3
        # ------------------------------------------------------------

        dinov3_result = (
            self.dinov3_checker.compare(
                reference_path,
                generated_path,
            )
        )

        # ------------------------------------------------------------
        # Object / environment
        # ------------------------------------------------------------

        object_environment_result = (
            self.object_environment_checker.compare(
                reference_path,
                generated_path,
            )
        )

        # ------------------------------------------------------------
        # Visual drift fallback
        # ------------------------------------------------------------

        fallback_result = None

        if (
            ssim_score < 0.75
            and lpips_score > 0.15
            and clip_score < 0.95
        ):
            fallback_score = (
                ssim_score
                + clip_score
                + (
                    1.0
                    - min(lpips_score, 1.0)
                )
            ) / 3.0

            fallback_result = {
                "label": "VISUAL_DRIFT_FALLBACK",
                "score": float(
                    fallback_score
                ),
            }

        return {
            "ssim": ssim_score,
            "lpips": lpips_score,
            "clip": clip_score,
            "dinov3": dinov3_result,
            "object_environment": (
                object_environment_result
            ),
            "visual_drift_fallback": (
                fallback_result
            ),
        }

    # ================================================================
    # Reason codes
    # ================================================================

    def _build_reason_codes(
        self,
        identity,
        visual,
    ):
        reason_codes = []

        # ------------------------------------------------------------
        # Identity
        # ------------------------------------------------------------

        identity_primary = identity["primary"]

        identity_label = (
            identity_primary["label"]
        )

        identity_reason_map = {
            "REFERENCE_FACE_MISSING":
                "ID_REFERENCE_FACE_MISSING",

            "GENERATED_FACE_MISSING":
                "ID_FACE_MISSING",

            "DIFFERENT_IDENTITY":
                "ID_FACE_MISMATCH",
        }

        if identity_label in identity_reason_map:
            self._add_reason(
                reason_codes,
                identity_reason_map[
                    identity_label
                ],
            )

        # ------------------------------------------------------------
        # SSIM
        # ------------------------------------------------------------

        if visual["ssim"] < 0.90:
            self._add_reason(
                reason_codes,
                "VISUAL_STRUCTURAL_CHANGE",
            )

        # ------------------------------------------------------------
        # LPIPS
        # ------------------------------------------------------------

        if visual["lpips"] > 0.30:
            self._add_reason(
                reason_codes,
                "VISUAL_PERCEPTUAL_CHANGE",
            )

        # ------------------------------------------------------------
        # CLIP
        # ------------------------------------------------------------

        if visual["clip"] < 0.85:
            self._add_reason(
                reason_codes,
                "VISUAL_LOW_SIMILARITY",
            )

        # ------------------------------------------------------------
        # DINOv3
        # ------------------------------------------------------------

        dinov3 = visual.get(
            "dinov3",
            {},
        )

        if dinov3.get("label") == "VISUAL_DRIFT":
            self._add_reason(
                reason_codes,
                dinov3.get(
                    "reason_code",
                    "VISUAL_DINOV3_DRIFT",
                ),
            )

        # ------------------------------------------------------------
        # Object / environment
        # ------------------------------------------------------------

        object_environment = visual.get(
            "object_environment",
            {},
        )

        object_result = object_environment.get(
            "object",
            {},
        )

        environment_result = (
            object_environment.get(
                "environment",
                {},
            )
        )

        if (
            object_result.get("label")
            == "OBJECT_DRIFT"
        ):
            self._add_reason(
                reason_codes,
                "VISUAL_OBJECT_DRIFT",
            )

        if (
            environment_result.get("label")
            == "ENVIRONMENT_DRIFT"
        ):
            self._add_reason(
                reason_codes,
                "VISUAL_ENVIRONMENT_DRIFT",
            )

        # ------------------------------------------------------------
        # Fallback
        # ------------------------------------------------------------

        fallback = visual.get(
            "visual_drift_fallback"
        )

        if fallback is not None:
            self._add_reason(
                reason_codes,
                "VISUAL_DRIFT_FALLBACK",
            )

        return reason_codes

    # ================================================================
    # Day 1 - Image analysis
    # ================================================================

    def analyze(
        self,
        reference_path,
        generated_path,
        output_path=None,
    ):
        """
        Analyze one reference image against one generated image.

        This preserves the Day 1 result structure.
        """

        reference_path = str(
            reference_path
        )

        generated_path = str(
            generated_path
        )

        # ------------------------------------------------------------
        # Identity
        # ------------------------------------------------------------

        identity = self._run_identity_checks(
            reference_path,
            generated_path,
        )

        # ------------------------------------------------------------
        # Visual
        # ------------------------------------------------------------

        visual = self._run_visual_checks(
            reference_path,
            generated_path,
        )

        # ------------------------------------------------------------
        # Reason codes
        # ------------------------------------------------------------

        reason_codes = (
            self._build_reason_codes(
                identity,
                visual,
            )
        )

        # ------------------------------------------------------------
        # Decision
        # ------------------------------------------------------------

        decision = (
            "PASS"
            if not reason_codes
            else "FAIL"
        )

        # ------------------------------------------------------------
        # Component scores
        # ------------------------------------------------------------

        component_scores = {
            "identity": identity,

            "adaface": identity[
                "primary"
            ],

            "ssim": visual[
                "ssim"
            ],

            "lpips": visual[
                "lpips"
            ],

            "clip": visual[
                "clip"
            ],

            "dinov3": visual[
                "dinov3"
            ],

            "object_environment": (
                visual[
                    "object_environment"
                ]
            ),
        }

        # ------------------------------------------------------------
        # Final result
        # ------------------------------------------------------------

        result = {
            "reference": reference_path,

            "generated": generated_path,

            "identity": identity,

            "visual": visual,

            "component_scores": (
                component_scores
            ),

            "decision": decision,

            "reason_codes": reason_codes,
        }

        # ------------------------------------------------------------
        # Optional JSON output
        # ------------------------------------------------------------

        if output_path:
            output_path = Path(
                output_path
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with output_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    result,
                    file,
                    indent=2,
                )

        return result

    # ================================================================
    # Day 2 - Deterministic video sampling
    # ================================================================

    def sample_video_frames(
        self,
        video_path,
        output_dir,
        sample_count=5,
    ):
        """
        Deterministically sample frames from an MP4.

        Example:

            25 frames
            5 samples

            [0, 6, 12, 18, 24]
        """

        video_path = Path(
            video_path
        )

        output_dir = Path(
            output_dir
        )

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: "
                f"{video_path}"
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():
            raise ValueError(
                f"Unable to open video: "
                f"{video_path}"
            )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        if total_frames <= 0:
            cap.release()

            raise ValueError(
                "Video contains no frames."
            )

        if sample_count <= 0:
            cap.release()

            raise ValueError(
                "sample_count must be "
                "greater than zero."
            )

        sample_count = min(
            sample_count,
            total_frames,
        )

        frame_indexes = np.linspace(
            0,
            total_frames - 1,
            sample_count,
            dtype=int,
        ).tolist()

        frame_indexes = sorted(
            set(frame_indexes)
        )

        extracted_frames = []

        current_index = 0

        while True:
            success, frame = cap.read()

            if not success:
                break

            if current_index in frame_indexes:

                frame_file = (
                    output_dir
                    / (
                        f"frame_"
                        f"{len(extracted_frames) + 1:02d}"
                        f"_index_"
                        f"{current_index:06d}.jpg"
                    )
                )

                success_write = cv2.imwrite(
                    str(frame_file),
                    frame,
                )

                if not success_write:
                    cap.release()

                    raise ValueError(
                        "Failed to write "
                        f"frame: {frame_file}"
                    )

                timestamp_s = (
                    current_index / fps
                    if fps > 0
                    else 0.0
                )

                extracted_frames.append(
                    {
                        "sample_number": (
                            len(
                                extracted_frames
                            )
                            + 1
                        ),

                        "frame_index": (
                            current_index
                        ),

                        "timestamp_s": float(
                            timestamp_s
                        ),

                        "file": str(
                            frame_file
                        ),
                    }
                )

            current_index += 1

        cap.release()

        if len(extracted_frames) != len(
            frame_indexes
        ):
            raise ValueError(
                "Unable to extract all "
                "requested video frames."
            )

        return {
            "total_frames": total_frames,

            "fps": fps,

            "sample_count": len(
                extracted_frames
            ),

            "frames": extracted_frames,
        }

    # ================================================================
    # ClipMetadata contract
    # ================================================================

    REQUIRED_CLIP_METADATA_FIELDS = {
        "clip_id",
        "shot_id",
        "mode",
        "source_type",
        "file",
        "duration_s",
        "fps",
        "width",
        "height",
        "aspect_ratio",
        "codec",
        "model",
        "settings",
        "reference_ids",
        "motion_controls",
        "seed",
        "status",
        "failure_reason",
        "checksum_sha256",
    }

    # ================================================================
    # Load ClipMetadata
    # ================================================================

    def _load_clip_metadata(
        self,
        metadata_path,
    ):
        """
        Load and validate ClipMetadata.

        The original shared contract fields
        remain unchanged.
        """

        metadata_path = Path(
            metadata_path
        )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"ClipMetadata file not found: "
                f"{metadata_path}"
            )

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        missing_fields = sorted(
            self.REQUIRED_CLIP_METADATA_FIELDS
            - set(metadata.keys())
        )

        if missing_fields:
            raise ValueError(
                "ClipMetadata is missing "
                "required fields: "
                f"{missing_fields}"
            )

        return metadata

    # ================================================================
    # SHA-256
    # ================================================================

    @staticmethod
    def _calculate_sha256(
        video_path,
    ):
        sha256 = hashlib.sha256()

        with open(
            video_path,
            "rb",
        ) as file:

            for chunk in iter(
                lambda: file.read(
                    1024 * 1024
                ),
                b"",
            ):
                sha256.update(chunk)

        return sha256.hexdigest()

    # ================================================================
    # Verify ClipMetadata
    # ================================================================

    def _verify_clip_metadata(
        self,
        metadata,
        video_path,
    ):
        """
        Verify ClipMetadata against the
        actual video file.
        """

        video_path = Path(
            video_path
        )

        # ------------------------------------------------------------
        # SHA-256
        # ------------------------------------------------------------

        actual_checksum = (
            self._calculate_sha256(
                video_path
            )
        )

        expected_checksum = str(
            metadata[
                "checksum_sha256"
            ]
        )

        checksum_verified = (
            actual_checksum.lower()
            == expected_checksum.lower()
        )

        # ------------------------------------------------------------
        # Open video
        # ------------------------------------------------------------

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():
            raise ValueError(
                f"Unable to open video: "
                f"{video_path}"
            )

        actual_frame_count = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        actual_fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        actual_width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        actual_height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        cap.release()

        # ------------------------------------------------------------
        # Expected values
        # ------------------------------------------------------------

        expected_frame_count = int(
            metadata[
                "settings"
            ][
                "frame_count"
            ]
        )

        expected_fps = float(
            metadata["fps"]
        )

        expected_width = int(
            metadata["width"]
        )

        expected_height = int(
            metadata["height"]
        )

        # ------------------------------------------------------------
        # Verification
        # ------------------------------------------------------------

        frame_count_verified = (
            actual_frame_count
            == expected_frame_count
        )

        fps_verified = (
            abs(
                actual_fps
                - expected_fps
            )
            < 1e-6
        )

        resolution_verified = (
            actual_width
            == expected_width
            and actual_height
            == expected_height
        )

        return {
            "checksum_verified": (
                checksum_verified
            ),

            "frame_count_verified": (
                frame_count_verified
            ),

            "fps_verified": (
                fps_verified
            ),

            "resolution_verified": (
                resolution_verified
            ),

            "actual_checksum": (
                actual_checksum
            ),

            "expected_checksum": (
                expected_checksum
            ),

            "actual_frame_count": (
                actual_frame_count
            ),

            "expected_frame_count": (
                expected_frame_count
            ),

            "actual_fps": (
                actual_fps
            ),

            "expected_fps": (
                expected_fps
            ),

            "actual_width": (
                actual_width
            ),

            "expected_width": (
                expected_width
            ),

            "actual_height": (
                actual_height
            ),

            "expected_height": (
                expected_height
            ),
        }

    # ================================================================
    # Day 2 clip decision
    # ================================================================

    def _determine_clip_decision(
        self,
        reason_codes,
    ):
        """
        Map QA reason codes to:

            PASS
            AUTO_RETRY
            HUMAN_REVIEW
        """

        auto_retry_reasons = {
            "ID_FACE_MISMATCH",
            "ID_FACE_MISSING",
            "ID_REFERENCE_FACE_MISSING",
            "VISUAL_DINOV3_DRIFT",
            "VISUAL_OBJECT_DRIFT",
            "VISUAL_ENVIRONMENT_DRIFT",
            "VISUAL_DRIFT_FALLBACK",
        }

        human_review_reasons = {
            "VISUAL_STRUCTURAL_CHANGE",
            "VISUAL_PERCEPTUAL_CHANGE",
            "VISUAL_LOW_SIMILARITY",
        }

        # ------------------------------------------------------------
        # PASS
        # ------------------------------------------------------------

        if not reason_codes:
            return "PASS"

        # ------------------------------------------------------------
        # AUTO_RETRY
        # ------------------------------------------------------------

        if any(
            reason in auto_retry_reasons
            for reason in reason_codes
        ):
            return "AUTO_RETRY"

        # ------------------------------------------------------------
        # HUMAN_REVIEW
        # ------------------------------------------------------------

        if any(
            reason in human_review_reasons
            for reason in reason_codes
        ):
            return "HUMAN_REVIEW"

        # ------------------------------------------------------------
        # Unknown failure
        # ------------------------------------------------------------

        return "HUMAN_REVIEW"

    # ================================================================
    # Day 2 - Analyze clip
    # ================================================================

    def analyze_clip(
        self,
        reference_path,
        video_path,
        output_dir,
        sample_count=5,
        clip_metadata_path=None,
        num_frames=None,
        qa_id=None,
        clip_id=None,
    ):
        """
        Analyze a reference image against deterministic
        sampled frames from an MP4 clip.

        Produces the Day 2 QAResult contract.
        """

        reference_path = str(
            reference_path
        )

        video_path = Path(
            video_path
        )

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ------------------------------------------------------------
        # num_frames overrides sample_count
        # ------------------------------------------------------------

        if num_frames is not None:
            sample_count = num_frames

        # ------------------------------------------------------------
        # Metadata
        # ------------------------------------------------------------

        metadata = None
        metadata_file = None
        verification = None

        if clip_metadata_path:

            metadata_file = Path(
                clip_metadata_path
            ).resolve()

            metadata = (
                self._load_clip_metadata(
                    metadata_file
                )
            )

            # --------------------------------------------------------
            # FIX 1:
            # metadata_file must be inside clip_metadata.
            # --------------------------------------------------------

            metadata[
                "metadata_file"
            ] = str(
                metadata_file
            )

            # --------------------------------------------------------
            # ClipMetadata clip_id is authoritative.
            # --------------------------------------------------------

            clip_id = metadata[
                "clip_id"
            ]

            # --------------------------------------------------------
            # Verify video against metadata.
            # --------------------------------------------------------

            verification = (
                self._verify_clip_metadata(
                    metadata,
                    video_path,
                )
            )

            # --------------------------------------------------------
            # FIX 2:
            # verification must also be inside clip_metadata.
            # --------------------------------------------------------

            metadata[
                "verification"
            ] = verification

        # ------------------------------------------------------------
        # Default clip ID
        # ------------------------------------------------------------

        if clip_id is None:
            clip_id = video_path.stem

        # ------------------------------------------------------------
        # QA ID
        # ------------------------------------------------------------

        if qa_id is None:
            qa_id = (
                f"qa_{clip_id}"
            )

        # ------------------------------------------------------------
        # Sample frames
        # ------------------------------------------------------------

        frames_dir = (
            output_dir
            / "frames"
        )

        sampling_result = (
            self.sample_video_frames(
                video_path=video_path,
                output_dir=frames_dir,
                sample_count=sample_count,
            )
        )

        frame_reports = []

        all_reason_codes = []

        # ============================================================
        # Analyze each sampled frame
        # ============================================================

        for frame_info in (
            sampling_result["frames"]
        ):

            frame_result = self.analyze(
                reference_path,
                frame_info["file"],
            )

            frame_reason_codes = (
                frame_result.get(
                    "reason_codes",
                    [],
                )
            )

            for reason in frame_reason_codes:

                self._add_reason(
                    all_reason_codes,
                    reason,
                )

            # --------------------------------------------------------
            # Complete frame-level evidence
            # --------------------------------------------------------

            frame_report = {
                "sample_number": (
                    frame_info[
                        "sample_number"
                    ]
                ),

                "frame_index": (
                    frame_info[
                        "frame_index"
                    ]
                ),

                "timestamp_s": (
                    frame_info[
                        "timestamp_s"
                    ]
                ),

                "file": (
                    frame_info[
                        "file"
                    ]
                ),

                "identity": (
                    frame_result[
                        "identity"
                    ]
                ),

                "visual": (
                    frame_result[
                        "visual"
                    ]
                ),

                "component_scores": (
                    frame_result[
                        "component_scores"
                    ]
                ),

                "decision": (
                    frame_result[
                        "decision"
                    ]
                ),

                "reason_codes": (
                    frame_reason_codes
                ),
            }

            frame_reports.append(
                frame_report
            )

        # ============================================================
        # Clip-level decision
        # ============================================================

        clip_decision = (
            self._determine_clip_decision(
                all_reason_codes
            )
        )

        # ============================================================
        # Sampling evidence
        # ============================================================

        sampled_frames = [
            {
                "frame_index": (
                    frame[
                        "frame_index"
                    ]
                ),

                "file": (
                    frame[
                        "file"
                    ]
                ),
            }

            for frame in (
                sampling_result[
                    "frames"
                ]
            )
        ]

        frames_sampled = [
            frame[
                "frame_index"
            ]

            for frame in (
                sampling_result[
                    "frames"
                ]
            )
        ]

        sampling = {
            "sample_count": (
                sampling_result[
                    "sample_count"
                ]
            ),

            "sampled_frames": (
                sampled_frames
            ),

            "frames_sampled": (
                frames_sampled
            ),
        }

        # ============================================================
        # Evidence files
        # ============================================================

        evidence_files = [
            frame["file"]
            for frame in frame_reports
        ]

        # ============================================================
        # QAResult
        # ============================================================

        result = {
            "qa_id": qa_id,

            "clip_id": clip_id,

            "qa_type": (
                "IDENTITY_VISUAL"
            ),

            "component_scores": {
                "frames": frame_reports,
            },

            "reason_codes": (
                all_reason_codes
            ),

            "decision": (
                clip_decision
            ),

            "evidence_files": (
                evidence_files
            ),

            "sampling": sampling,
        }

        # ============================================================
        # Add ClipMetadata
        # ============================================================

        if metadata is not None:

            # IMPORTANT:
            # metadata already contains:
            #
            # metadata_file
            # verification
            #
            # because they were added above.

            result[
                "clip_metadata"
            ] = metadata

            # Keep top-level compatibility fields too.

            result[
                "metadata_file"
            ] = str(
                metadata_file
            )

            result[
                "verification"
            ] = verification

        # ============================================================
        # QAResult JSON path
        # ============================================================

        qa_result_path = (
            output_dir
            / (
                f"{clip_id}"
                "_QAResult.json"
            )
        )

        # ------------------------------------------------------------
        # Include QAResult JSON itself as evidence
        # ------------------------------------------------------------

        result[
            "evidence_files"
        ].append(
            str(
                qa_result_path
            )
        )

        # ============================================================
        # Save QAResult
        # ============================================================

        with qa_result_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                indent=2,
            )

        return result


# ====================================================================
# Standalone check
# ====================================================================

if __name__ == "__main__":

    engine = QAEngine()

    print(
        "QAEngine loaded successfully."
    )