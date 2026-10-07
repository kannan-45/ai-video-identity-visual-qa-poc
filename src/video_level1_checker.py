from pathlib import Path

import cv2
import numpy as np


class VideoLevel1Checker:
    """
    Level 1 cheap video checks.

    CPU-friendly checks over decoded video frames before
    expensive identity and visual models are executed.

    Checks:
        1. Coarse deterministic frame sampling
        2. Frame-to-frame difference
        3. Brightness delta
        4. Color delta
        5. Frozen-frame run detection
        6. Motion energy
    """

    def __init__(
        self,
        frame_diff_threshold=0.08,
        brightness_delta_threshold=0.15,
        color_delta_threshold=0.15,
        frozen_frame_threshold=0.001,
        motion_energy_threshold=0.001,
        frozen_run_length=3,
    ):
        self.frame_diff_threshold = frame_diff_threshold
        self.brightness_delta_threshold = brightness_delta_threshold
        self.color_delta_threshold = color_delta_threshold
        self.frozen_frame_threshold = frozen_frame_threshold
        self.motion_energy_threshold = motion_energy_threshold
        self.frozen_run_length = frozen_run_length

    def get_video_info(self, video_path):
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(
                f"Unable to open video: {video_path}"
            )

        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = float(
            cap.get(cv2.CAP_PROP_FPS)
        )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        duration_s = (
            frame_count / fps
            if fps > 0
            else 0.0
        )

        cap.release()

        return {
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_s": duration_s,
        }

    def decode_all_frames(self, video_path):
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(
                f"Unable to open video: {video_path}"
            )

        frames = []

        while True:
            success, frame = cap.read()

            if not success:
                break

            frames.append(frame)

        cap.release()

        return frames

    def sample_frames(
        self,
        total_frames,
        sample_count,
    ):
        if total_frames <= 0:
            return []

        if sample_count <= 0:
            raise ValueError(
                "sample_count must be greater than zero"
            )

        sample_count = min(
            sample_count,
            total_frames,
        )

        indexes = np.linspace(
            0,
            total_frames - 1,
            sample_count,
            dtype=int,
        )

        return indexes.tolist()

    @staticmethod
    def _resize_for_analysis(frame):
        return cv2.resize(
            frame,
            (160, 120),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _brightness(frame):
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        return float(
            np.mean(gray) / 255.0
        )

    @staticmethod
    def _color_mean(frame):
        resized = (
            VideoLevel1Checker
            ._resize_for_analysis(frame)
        )

        mean_color = np.mean(
            resized,
            axis=(0, 1),
        )

        return mean_color / 255.0

    def calculate_frame_difference(
        self,
        previous_frame,
        current_frame,
    ):
        previous = (
            self._resize_for_analysis(
                previous_frame
            )
        )

        current = (
            self._resize_for_analysis(
                current_frame
            )
        )

        previous_gray = cv2.cvtColor(
            previous,
            cv2.COLOR_BGR2GRAY,
        )

        current_gray = cv2.cvtColor(
            current,
            cv2.COLOR_BGR2GRAY,
        )

        difference = cv2.absdiff(
            previous_gray,
            current_gray,
        )

        return float(
            np.mean(difference) / 255.0
        )

    def calculate_brightness_delta(
        self,
        previous_frame,
        current_frame,
    ):
        previous = self._brightness(
            previous_frame
        )

        current = self._brightness(
            current_frame
        )

        return abs(
            current - previous
        )

    def calculate_color_delta(
        self,
        previous_frame,
        current_frame,
    ):
        previous = self._color_mean(
            previous_frame
        )

        current = self._color_mean(
            current_frame
        )

        return float(
            np.mean(
                np.abs(
                    current - previous
                )
            )
        )

    def calculate_motion_energy(
        self,
        previous_frame,
        current_frame,
    ):
        return self.calculate_frame_difference(
            previous_frame,
            current_frame,
        )

    def is_low_difference(
        self,
        frame_difference,
    ):
        return (
            frame_difference
            <= self.frozen_frame_threshold
        )

    def _detect_frozen_runs(
        self,
        frame_scan,
    ):
        """
        Detect consecutive near-identical frame pairs.

        A single low-difference pair is not treated
        as a frozen frame. A frozen condition requires
        multiple consecutive low-difference pairs.
        """

        frozen_frames = set()

        run = []

        for item in frame_scan:
            if item["frame_difference"] <= (
                self.frozen_frame_threshold
            ):
                run.append(
                    item["frame_index"]
                )

                if len(run) >= self.frozen_run_length:
                    frozen_frames.update(run)
            else:
                run = []

        return frozen_frames

    def scan_video(
        self,
        video_path,
        sample_count=5,
    ):
        video_info = self.get_video_info(
            video_path
        )

        frames = self.decode_all_frames(
            video_path
        )

        actual_frame_count = len(frames)

        results = []

        for index in range(
            1,
            actual_frame_count,
        ):
            previous_frame = frames[index - 1]
            current_frame = frames[index]

            frame_difference = (
                self.calculate_frame_difference(
                    previous_frame,
                    current_frame,
                )
            )

            brightness_delta = (
                self.calculate_brightness_delta(
                    previous_frame,
                    current_frame,
                )
            )

            color_delta = (
                self.calculate_color_delta(
                    previous_frame,
                    current_frame,
                )
            )

            motion_energy = (
                self.calculate_motion_energy(
                    previous_frame,
                    current_frame,
                )
            )

            results.append(
                {
                    "frame_index": index,
                    "previous_frame_index": index - 1,
                    "frame_difference": frame_difference,
                    "brightness_delta": brightness_delta,
                    "color_delta": color_delta,
                    "motion_energy": motion_energy,
                    "frozen_frame": False,
                    "anomaly_candidate": False,
                    "anomaly_reasons": [],
                }
            )

        frozen_frames = self._detect_frozen_runs(
            results
        )

        anomaly_candidate_frames = []

        for item in results:
            index = item["frame_index"]

            anomaly_reasons = []

            if (
                item["frame_difference"]
                >= self.frame_diff_threshold
            ):
                anomaly_reasons.append(
                    "FRAME_DIFF_SPIKE"
                )

            if (
                item["brightness_delta"]
                >= self.brightness_delta_threshold
            ):
                anomaly_reasons.append(
                    "BRIGHTNESS_DELTA"
                )

            if (
                item["color_delta"]
                >= self.color_delta_threshold
            ):
                anomaly_reasons.append(
                    "COLOR_DELTA"
                )

            if index in frozen_frames:
                anomaly_reasons.append(
                    "FROZEN_FRAME"
                )

            # Low motion alone is not considered an
            # anomaly. It becomes useful when combined
            # with a detected frozen run.
            if (
                index in frozen_frames
                and item["motion_energy"]
                <= self.motion_energy_threshold
            ):
                anomaly_reasons.append(
                    "LOW_MOTION_ENERGY"
                )

            is_candidate = (
                len(anomaly_reasons) > 0
            )

            item["frozen_frame"] = (
                index in frozen_frames
            )

            item["anomaly_candidate"] = (
                is_candidate
            )

            item["anomaly_reasons"] = (
                anomaly_reasons
            )

            if is_candidate:
                anomaly_candidate_frames.append(
                    index
                )

        coarse_frame_indexes = (
            self.sample_frames(
                actual_frame_count,
                sample_count,
            )
        )

        return {
            "video": {
                "file": str(
                    Path(video_path)
                ),
                **video_info,
                "decoded_frame_count":
                    actual_frame_count,
            },
            "coarse_frame_indexes":
                coarse_frame_indexes,
            "anomaly_candidate_frames":
                anomaly_candidate_frames,
            "frame_scan":
                results,
        }