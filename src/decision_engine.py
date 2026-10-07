class DecisionEngine:
    """
    Converts aggregated video QA evidence into the
    final clip-level decision.

    Decision mapping:
        PASS
        AUTO_RETRY
        HUMAN_REVIEW
    """

    AUTO_RETRY_REASONS = {
        "ID_FACE_MISMATCH",
        "ID_FACE_MISSING",
        "ID_REFERENCE_FACE_MISSING",
        "VISUAL_DINOV3_DRIFT",
        "VISUAL_OBJECT_DRIFT",
        "VISUAL_ENVIRONMENT_DRIFT",
        "VISUAL_DRIFT_FALLBACK",
    }

    HUMAN_REVIEW_REASONS = {
        "VISUAL_STRUCTURAL_CHANGE",
        "VISUAL_PERCEPTUAL_CHANGE",
        "VISUAL_LOW_SIMILARITY",
    }

    def decide(self, evidence):
        reason_codes = evidence["aggregate"].get(
            "reason_codes",
            [],
        )

        if not reason_codes:
            return {
                "decision": "PASS",
                "reason_codes": [],
                "explanation": (
                    "No QA failure reason codes were "
                    "detected."
                ),
            }

        auto_retry_reasons = [
            reason
            for reason in reason_codes
            if reason in self.AUTO_RETRY_REASONS
        ]

        human_review_reasons = [
            reason
            for reason in reason_codes
            if reason in self.HUMAN_REVIEW_REASONS
        ]

        if auto_retry_reasons:
            return {
                "decision": "AUTO_RETRY",
                "reason_codes": reason_codes,
                "explanation": (
                    "One or more high-confidence "
                    "identity or severe visual failure "
                    "conditions require automatic retry."
                ),
            }

        if human_review_reasons:
            return {
                "decision": "HUMAN_REVIEW",
                "reason_codes": reason_codes,
                "explanation": (
                    "Visual changes were detected and "
                    "require human review."
                ),
            }

        return {
            "decision": "HUMAN_REVIEW",
            "reason_codes": reason_codes,
            "explanation": (
                "An unclassified QA failure was detected "
                "and requires human review."
            ),
        }