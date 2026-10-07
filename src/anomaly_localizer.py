class AnomalyLocalizer:
    """
    Converts frame-level Level 2 failures into
    contiguous flagged frame ranges.
    """

    def __init__(self, max_gap=1):
        self.max_gap = max_gap

    def localize(self, frame_results):
        """
        Extract failed frame indexes and group nearby
        failures into contiguous ranges.
        """

        failed_frames = []

        for result in frame_results:
            qa_result = result["qa_result"]

            if qa_result["decision"] != "PASS":
                failed_frames.append(
                    result["frame_index"]
                )

        failed_frames = sorted(
            set(failed_frames)
        )

        if not failed_frames:
            return {
                "flagged_frames": [],
                "flagged_ranges": [],
            }

        ranges = []

        start = failed_frames[0]
        end = failed_frames[0]

        for frame_index in failed_frames[1:]:
            if frame_index - end <= self.max_gap:
                end = frame_index
            else:
                ranges.append(
                    {
                        "start": start,
                        "end": end,
                    }
                )

                start = frame_index
                end = frame_index

        ranges.append(
            {
                "start": start,
                "end": end,
            }
        )

        return {
            "flagged_frames": failed_frames,
            "flagged_ranges": ranges,
        }