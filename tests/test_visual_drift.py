from pathlib import Path

from src.qa_engine import QAEngine


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REFERENCE = (
    PROJECT_ROOT
    / "mock_data"
    / "inputs"
    / "adaface"
    / "reference"
    / "reference_adaface.jpeg"
)

VISUAL_DRIFT_FIXTURE = (
    PROJECT_ROOT
    / "mock_data"
    / "inputs"
    / "adaface"
    / "generated"
    / "generated_visual_drift.jpeg"
)


def test_visual_drift_fixture_exists():
    assert REFERENCE.exists(), (
        f"Reference fixture not found: {REFERENCE}"
    )

    assert VISUAL_DRIFT_FIXTURE.exists(), (
        f"Visual drift fixture not found: {VISUAL_DRIFT_FIXTURE}"
    )


def test_visual_drift_is_detected():
    engine = QAEngine()

    result = engine.analyze(
        str(REFERENCE),
        str(VISUAL_DRIFT_FIXTURE)
    )

    assert result["decision"] == "FAIL"

    reason_codes = result["reason_codes"]

    # The visual-drift fixture must produce at least
    # one explicit visual-failure reason.
    visual_failure_codes = {
        "VISUAL_STRUCTURAL_CHANGE",
        "VISUAL_PERCEPTUAL_CHANGE",
        "VISUAL_LOW_SIMILARITY",
        "VISUAL_DINOV3_DRIFT",
        "VISUAL_DRIFT_FALLBACK",
        "VISUAL_OBJECT_DRIFT",
        "VISUAL_ENVIRONMENT_DRIFT",
    }

    assert visual_failure_codes.intersection(reason_codes), (
        f"No visual failure reason detected. "
        f"Reason codes: {reason_codes}"
    )


def test_visual_drift_has_component_evidence():
    engine = QAEngine()

    result = engine.analyze(
        str(REFERENCE),
        str(VISUAL_DRIFT_FIXTURE)
    )

    visual = result["visual"]

    assert "ssim" in visual
    assert "lpips" in visual
    assert "clip" in visual
    assert "dinov3" in visual
    assert "object_environment" in visual

    assert visual["ssim"] < 0.75


def test_dinov3_status_is_reported():
    engine = QAEngine()

    result = engine.analyze(
        str(REFERENCE),
        str(VISUAL_DRIFT_FIXTURE)
    )

    dinov3 = result["visual"]["dinov3"]

    assert "available" in dinov3
    assert "score" in dinov3
    assert "label" in dinov3

def test_coco_object_drift():
    reference = (
        PROJECT_ROOT
        / "mock_data"
        / "inputs"
        / "coco"
        / "images"
        / "000000000139.jpg"
    )

    generated = (
        PROJECT_ROOT
        / "mock_data"
        / "inputs"
        / "coco"
        / "failures"
        / "000000000139_object_change.jpg"
    )

    result = QAEngine().analyze(str(reference), str(generated))

    assert result["decision"] == "FAIL"

    assert "VISUAL_OBJECT_DRIFT" in result["reason_codes"]

    object_result = result["visual"]["object_environment"]["object"]

    assert object_result["label"] == "OBJECT_DRIFT"
    assert object_result["score"] < 0.70

    environment_result = result["visual"]["object_environment"]["environment"]

    assert environment_result["label"] == "ENVIRONMENT_CONSISTENT"