class DenseResampler:
    """
    Creates additional frame indexes around flagged
    anomaly ranges for a second, denser QA pass.
    """

    def __init__(self, padding=1):
        self.padding = padding

    def dense_resample(
        self,
        flagged_ranges,
        total_frames,
    ):
        """
        Add neighboring frames around every flagged range.

        Frames are clamped to the valid video range and
        duplicates are removed.
        """

        if total_frames <= 0:
            return []

        selected = set()

        for flagged_range in flagged_ranges:
            start = int(flagged_range["start"])
            end = int(flagged_range["end"])

            dense_start = max(
                0,
                start - self.padding,
            )

            dense_end = min(
                total_frames - 1,
                end + self.padding,
            )

            for frame_index in range(
                dense_start,
                dense_end + 1,
            ):
                selected.add(frame_index)

        return sorted(selected)