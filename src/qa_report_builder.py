from datetime import datetime, timezone


class QAReportBuilder:
    """
    Builds the final video QA report from:
    - Clip metadata
    - Level 1 evidence
    - Level 2 first pass
    - Anomaly localization
    - Dense second pass
    - EvidenceVector
    - Final decision
    """

    def build(
        self,
        clip_id,
        clip_metadata,
        evidence,
        decision,
    ):
        return {
            "qa_id": f"qa_{clip_id}",
            "clip_id": clip_id,
            "qa_type": "IDENTITY_VISUAL",
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "decision": decision["decision"],
            "reason_codes": decision["reason_codes"],
            "explanation": decision["explanation"],
            "clip_metadata": clip_metadata,
            "evidence": evidence,
        }