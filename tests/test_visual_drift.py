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
    """Verify the deliberate visual-drift fixture exists."""

    assert REFERENCE.exists(), (
        f"Reference fixture not found: {REFERENCE}"
    )

    assert VISUAL_DRIFT_FIXTURE.exists(), (
        f"Visual drift fixture not found: "
        f"{VISUAL_DRIFT_FIXTURE}"
    )


def test_visual_drift_is_detected():
    """
    Verify that the deliberate visual-drift fixture
    produces a FAIL decision and visual-drift evidence.
    """

    engine = QAEngine()

    result = engine.analyze(
        str(REFERENCE),
        str(VISUAL_DRIFT_FIXTURE)
    )

    assert result["decision"] == "FAIL"

    reason_codes = result["reason_codes"]

    assert (
        "VISUAL_DRIFT_FALLBACK" in reason_codes
        or "VISUAL_DINOV3_DRIFT" in reason_codes
    )


def test_visual_drift_has_component_evidence():
    """
    Verify that visual drift is supported by component-level
    evidence rather than a single opaque score.
    """

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

    # Our current deliberate fixture should show
    # substantial structural change.
    assert visual["ssim"] < 0.75