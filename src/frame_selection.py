from typing import Iterable, List


def merge_frame_selection(
    coarse_frame_indexes: Iterable[int],
    anomaly_candidate_frames: Iterable[int],
) -> List[int]:
    """
    Merge deterministic coarse samples with Level 1
    anomaly candidates.

    Rules:
        1. Keep every coarse frame.
        2. Add every anomaly candidate.
        3. Remove duplicates.
        4. Return sorted frame indexes.
    """

    coarse = set(
        int(index)
        for index in coarse_frame_indexes
    )

    candidates = set(
        int(index)
        for index in anomaly_candidate_frames
    )

    merged = coarse.union(candidates)

    return sorted(merged)