from pathlib import Path

import cv2


class Level2FrameRunner:
    """
    Level 2 expensive video QA runner.

    Level 1 decides WHICH frames need expensive analysis.
    This runner extracts those frames and sends each one
    through the existing Day 1 QAEngine.
    """

    def __init__(self, qa_engine):
        self.qa_engine = qa_engine

    def extract_selected_frames(
        self,
        video_path,
        frame_indexes,
        output_dir,
    ):
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        requested = sorted(
            set(int(i) for i in frame_indexes)
        )

        requested_set = set(requested)

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():
            raise ValueError(
                f"Unable to open video: {video_path}"
            )

        extracted = {}

        current_index = 0

        while True:
            success, frame = cap.read()

            if not success:
                break

            if current_index in requested_set:
                frame_path = (
                    output_dir
                    / f"frame_{current_index:06d}.jpg"
                )

                cv2.imwrite(
                    str(frame_path),
                    frame,
                )

                extracted[current_index] = (
                    frame_path
                )

            current_index += 1

        cap.release()

        missing = [
            index
            for index in requested
            if index not in extracted
        ]

        if missing:
            raise ValueError(
                "Requested frame indexes were not "
                f"found in video: {missing}"
            )

        return extracted

    def analyze_selected_frames(
        self,
        reference_path,
        video_path,
        frame_indexes,
        output_dir,
    ):
        """
        Extract and analyze selected frames using
        the existing Day 1 QAEngine.
        """

        if self.qa_engine is None:
            raise ValueError(
                "qa_engine is required for Level 2 analysis."
            )

        extracted = self.extract_selected_frames(
            video_path=video_path,
            frame_indexes=frame_indexes,
            output_dir=output_dir,
        )

        frame_results = []

        for frame_index in sorted(extracted):
            frame_path = extracted[
                frame_index
            ]

            qa_result = self.qa_engine.analyze(
                reference_path,
                str(frame_path),
            )

            frame_results.append(
                {
                    "frame_index": frame_index,
                    "frame_file": str(frame_path),
                    "qa_result": qa_result,
                }
            )

        return frame_results