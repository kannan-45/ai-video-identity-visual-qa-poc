class EvidenceAggregator:
    """
    Aggregates Level 1 and Level 2 evidence into a single
    EvidenceVector for the final decision engine.
    """

    def aggregate(
        self,
        level1_result,
        first_pass_results,
        flagged_ranges,
        dense_pass_results,
    ):
        all_results = (
            list(first_pass_results)
            + list(dense_pass_results)
        )

        decisions = []
        reason_codes = []
        failed_frames = []

        for result in all_results:
            qa_result = result["qa_result"]

            decisions.append(
                qa_result["decision"]
            )

            for reason in qa_result.get(
                "reason_codes",
                [],
            ):
                if reason not in reason_codes:
                    reason_codes.append(reason)

            if qa_result["decision"] != "PASS":
                failed_frames.append(
                    result["frame_index"]
                )

        failed_frames = sorted(
            set(failed_frames)
        )

        return {
            "evidence_type": "IDENTITY_VISUAL_VIDEO",
            "level1": {
                "coarse_frame_indexes": level1_result[
                    "coarse_frame_indexes"
                ],
                "anomaly_candidate_frames": level1_result[
                    "anomaly_candidate_frames"
                ],
                "decoded_frame_count": level1_result[
                    "video"
                ]["decoded_frame_count"],
            },
            "first_pass": {
                "frames_analyzed": len(
                    first_pass_results
                ),
                "failed_frames": sorted(
                    set(
                        result["frame_index"]
                        for result in first_pass_results
                        if result["qa_result"]["decision"]
                        != "PASS"
                    )
                ),
            },
            "anomaly_ranges": flagged_ranges,
            "dense_pass": {
                "frames_analyzed": len(
                    dense_pass_results
                ),
                "failed_frames": sorted(
                    set(
                        result["frame_index"]
                        for result in dense_pass_results
                        if result["qa_result"]["decision"]
                        != "PASS"
                    )
                ),
            },
            "aggregate": {
                "total_frames_checked": len(
                    all_results
                ),
                "failed_frames": failed_frames,
                "reason_codes": reason_codes,
                "decisions": sorted(
                    set(decisions)
                ),
            },
        }