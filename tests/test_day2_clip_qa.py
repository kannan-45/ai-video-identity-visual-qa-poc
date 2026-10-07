import json
from pathlib import Path

from src.qa_engine import QAEngine


ROOT = Path(__file__).resolve().parents[1]

REFERENCE = (
    ROOT
    / "mock_data"
    / "inputs"
    / "adaface"
    / "reference"
    / "reference_adaface.jpeg"
)

CLEAN_VIDEO = (
    ROOT
    / "mock_data"
    / "inputs"
    / "clean_test_video.mp4"
)

FAILURE_VIDEO = (
    ROOT
    / "mock_data"
    / "inputs"
    / "test_video.mp4"
)

CLIP_METADATA = (
    ROOT
    / "mock_data"
    / "inputs"
    / "member7"
    / "clip_pan_zoom_001.json"
)

CLIP_METADATA_VIDEO = (
    ROOT
    / "mock_data"
    / "inputs"
    / "member7"
    / "clip_pan_zoom_001.mp4"
)


def test_clean_clip_returns_pass(tmp_path):
    engine = QAEngine()

    result = engine.analyze_clip(
        REFERENCE,
        CLEAN_VIDEO,
        output_dir=tmp_path,
        num_frames=5,
        qa_id="test_day2_clean",
        clip_id="test_clip_clean",
    )

    assert result["decision"] == "PASS"
    assert result["reason_codes"] == []

    sampled_indexes = [
        frame["frame_index"]
        for frame in result["sampling"]["sampled_frames"]
    ]

    assert sampled_indexes == [0, 6, 12, 18, 24]


def test_failure_clip_returns_auto_retry(tmp_path):
    engine = QAEngine()

    result = engine.analyze_clip(
        REFERENCE,
        FAILURE_VIDEO,
        output_dir=tmp_path,
        num_frames=5,
        qa_id="test_day2_failure",
        clip_id="test_clip_failure",
    )

    assert result["decision"] == "AUTO_RETRY"

    assert len(result["reason_codes"]) > 0


def test_clip_sampling_is_deterministic(tmp_path):
    engine = QAEngine()

    result1 = engine.analyze_clip(
        REFERENCE,
        CLEAN_VIDEO,
        output_dir=tmp_path / "run1",
        num_frames=5,
        qa_id="test_sampling_1",
        clip_id="sampling_clip_1",
    )

    result2 = engine.analyze_clip(
        REFERENCE,
        CLEAN_VIDEO,
        output_dir=tmp_path / "run2",
        num_frames=5,
        qa_id="test_sampling_2",
        clip_id="sampling_clip_2",
    )

    indexes1 = [
        frame["frame_index"]
        for frame in result1["sampling"]["sampled_frames"]
    ]

    indexes2 = [
        frame["frame_index"]
        for frame in result2["sampling"]["sampled_frames"]
    ]

    assert indexes1 == indexes2
    assert indexes1 == [0, 6, 12, 18, 24]


def test_clip_result_contains_component_evidence(tmp_path):
    engine = QAEngine()

    result = engine.analyze_clip(
        REFERENCE,
        CLEAN_VIDEO,
        output_dir=tmp_path,
        num_frames=5,
        qa_id="test_evidence",
        clip_id="test_clip_evidence",
    )

    assert result["qa_type"] == "IDENTITY_VISUAL"

    assert "component_scores" in result
    assert "frames" in result["component_scores"]

    frames = result["component_scores"]["frames"]

    assert len(frames) == 5

    first_frame = frames[0]

    assert "identity" in first_frame
    assert "visual" in first_frame


def test_qa_result_json_is_created(tmp_path):
    engine = QAEngine()

    result = engine.analyze_clip(
        REFERENCE,
        CLEAN_VIDEO,
        output_dir=tmp_path,
        num_frames=5,
        qa_id="test_json",
        clip_id="test_clip_json",
    )

    qa_files = [
        Path(file)
        for file in result["evidence_files"]
        if str(file).endswith("_QAResult.json")
    ]

    assert len(qa_files) == 1
    assert qa_files[0].exists()

    with open(qa_files[0], "r", encoding="utf-8") as f:
        saved_result = json.load(f)

    assert saved_result["qa_type"] == "IDENTITY_VISUAL"
    assert saved_result["decision"] == "PASS"


def test_clip_metadata_is_consumed_and_verified(tmp_path):
    engine = QAEngine()

    result = engine.analyze_clip(
        REFERENCE,
        CLIP_METADATA_VIDEO,
        output_dir=tmp_path,
        num_frames=5,
        qa_id="test_clip_metadata",
        clip_id="ignored_clip_id",
        clip_metadata_path=CLIP_METADATA,
    )

    # ------------------------------------------------------------
    # QAResult contract
    # ------------------------------------------------------------

    assert result["qa_type"] == "IDENTITY_VISUAL"

    assert result["clip_id"] == "clip_pan_zoom_001"

    # ------------------------------------------------------------
    # ClipMetadata must be included
    # ------------------------------------------------------------

    assert "clip_metadata" in result

    metadata = result["clip_metadata"]

    # ------------------------------------------------------------
    # Core ClipMetadata fields
    # ------------------------------------------------------------

    assert metadata["clip_id"] == "clip_pan_zoom_001"

    assert metadata["shot_id"] == "shot_member7_day2_001"

    assert metadata["mode"] == "I2V"

    assert metadata["source_type"] == "mock"

    assert (
        metadata["file"]
        == "mock_data/inputs/member7/clip_pan_zoom_001.mp4"
    )

    # ------------------------------------------------------------
    # Video properties
    # ------------------------------------------------------------

    assert metadata["duration_s"] == 5.0

    assert metadata["fps"] == 5.0

    assert metadata["width"] == 260

    assert metadata["height"] == 194

    assert metadata["aspect_ratio"] == "130:97"

    assert metadata["codec"] == "h264"

    # ------------------------------------------------------------
    # Model / settings
    # ------------------------------------------------------------

    assert metadata["model"] == "mock-ffmpeg-zoompan"

    assert metadata["settings"]["frame_count"] == 25

    assert metadata["settings"]["sampling_count"] == 5

    # ------------------------------------------------------------
    # Reference and motion metadata
    # ------------------------------------------------------------

    assert metadata["reference_ids"] == [
        "reference_adaface"
    ]

    assert metadata["motion_controls"]["type"] == "zoom"

    assert metadata["motion_controls"]["deterministic"] is True

    assert metadata["motion_controls"]["zoom_increment"] == 0.02

    assert metadata["motion_controls"]["maximum_zoom"] == 1.5

    # ------------------------------------------------------------
    # Status fields
    # ------------------------------------------------------------

    assert metadata["seed"] is None

    assert metadata["status"] == "SUCCESS"

    assert metadata["failure_reason"] is None

    # ------------------------------------------------------------
    # SHA-256 checksum
    # ------------------------------------------------------------

    assert (
        metadata["checksum_sha256"]
        == "E24AFEDB7CB8452EEAEABF92D58E80E8CF131A7F38477F5FA77E2932BFAB2476"
    )

    # ------------------------------------------------------------
    # Additional QA metadata
    # ------------------------------------------------------------

    assert "metadata_file" in metadata

    assert (
        Path(metadata["metadata_file"]).resolve()
        == CLIP_METADATA.resolve()
    )

    # ------------------------------------------------------------
    # ClipMetadata verification
    # ------------------------------------------------------------

    assert "verification" in metadata

    verification = metadata["verification"]

    assert verification["checksum_verified"] is True

    assert verification["frame_count_verified"] is True

    assert verification["fps_verified"] is True

    assert verification["resolution_verified"] is True

    # ------------------------------------------------------------
    # Deterministic sampling
    # ------------------------------------------------------------

    sampled_indexes = [
        frame["frame_index"]
        for frame in result["sampling"]["sampled_frames"]
    ]

    assert sampled_indexes == [0, 6, 12, 18, 24]

    # ------------------------------------------------------------
    # Component evidence must still be retained
    # ------------------------------------------------------------

    assert "component_scores" in result

    assert "frames" in result["component_scores"]

    assert len(result["component_scores"]["frames"]) == 5