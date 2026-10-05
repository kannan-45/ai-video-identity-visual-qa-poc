import cv2
import os
import json
import argparse


class KeyframeQA:

    def __init__(self, qa_engine):
        self.qa_engine = qa_engine

    def extract_keyframes(
        self,
        video_path,
        output_dir,
        num_frames=5
    ):
        """
        Extract evenly spaced keyframes from a video.
        """

        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        os.makedirs(output_dir, exist_ok=True)

        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():
            raise ValueError(
                f"Could not open video: {video_path}"
            )

        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = capture.get(cv2.CAP_PROP_FPS)

        if total_frames <= 0:
            capture.release()
            raise ValueError(
                "Video contains no frames."
            )

        num_frames = min(
            num_frames,
            total_frames
        )

        frame_indices = [
            int(i * (total_frames - 1) / (num_frames - 1))
            if num_frames > 1
            else 0
            for i in range(num_frames)
        ]

        keyframes = []

        for index, frame_number in enumerate(
            frame_indices
        ):

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number
            )

            success, frame = capture.read()

            if not success:
                continue

            output_path = os.path.join(
                output_dir,
                f"keyframe_{index + 1:02d}.jpg"
            )

            cv2.imwrite(
                output_path,
                frame
            )

            timestamp = (
                frame_number / fps
                if fps > 0
                else None
            )

            keyframes.append({
                "frame_number": frame_number,
                "timestamp_seconds": (
                    round(timestamp, 3)
                    if timestamp is not None
                    else None
                ),
                "path": output_path
            })

        capture.release()

        return keyframes

    def analyze_video(
        self,
        reference_path,
        video_path,
        output_dir,
        num_frames=5
    ):
        """
        Extract keyframes and compare every frame
        against the reference image.
        """

        keyframes = self.extract_keyframes(
            video_path,
            output_dir,
            num_frames
        )

        results = []

        for keyframe in keyframes:

            print(
                f"\nAnalyzing frame "
                f"{keyframe['frame_number']} "
                f"at "
                f"{keyframe['timestamp_seconds']}s..."
            )

            qa_result = self.qa_engine.analyze(
                reference_path,
                keyframe["path"]
            )

            results.append({
                "frame_number": (
                    keyframe["frame_number"]
                ),
                "timestamp_seconds": (
                    keyframe["timestamp_seconds"]
                ),
                "path": keyframe["path"],
                "decision": (
                    qa_result["decision"]
                ),
                "reason_codes": (
                    qa_result["reason_codes"]
                ),
                "identity": (
                    qa_result["identity"]
                ),
                "visual": (
                    qa_result["visual"]
                )
            })

        failed_frames = [
            result
            for result in results
            if result["decision"] == "FAIL"
        ]

        overall_decision = (
            "FAIL"
            if failed_frames
            else "PASS"
        )

        return {
            "video": video_path,
            "reference": reference_path,
            "frames_checked": len(results),
            "failed_frames": len(failed_frames),
            "overall_decision": overall_decision,
            "frames": results
        }


def main():

    parser = argparse.ArgumentParser(
        description="Keyframe QA for AI-generated videos"
    )

    parser.add_argument(
        "--reference",
        default=(
            "mock_data/inputs/adaface/reference/"
            "reference_adaface.jpeg"
        ),
        help="Reference image path"
    )

    parser.add_argument(
        "--video",
        default="mock_data/inputs/test_video.mp4",
        help="Video path"
    )

    parser.add_argument(
        "--output",
        default="mock_data/outputs/keyframes",
        help="Directory for extracted keyframes"
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=5,
        help="Number of evenly spaced keyframes"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("KEYFRAME QA TEST")
    print("=" * 60)

    print(f"Reference : {args.reference}")
    print(f"Video     : {args.video}")
    print(f"Frames    : {args.num_frames}")

    from src.qa_engine import QAEngine

    engine = QAEngine()

    keyframe_qa = KeyframeQA(engine)

    report = keyframe_qa.analyze_video(
        reference_path=args.reference,
        video_path=args.video,
        output_dir=args.output,
        num_frames=args.num_frames
    )

    print("\n" + "=" * 60)
    print("KEYFRAME QA RESULT")
    print("=" * 60)

    print(
        f"Frames checked : "
        f"{report['frames_checked']}"
    )

    print(
        f"Failed frames  : "
        f"{report['failed_frames']}"
    )

    print(
        f"Overall result : "
        f"{report['overall_decision']}"
    )

    print("\nFrame Results:")

    for frame in report["frames"]:

        print(
            f"  Frame {frame['frame_number']:>3} "
            f"| {frame['timestamp_seconds']}s "
            f"| {frame['decision']}"
        )

        if frame["reason_codes"]:

            print(
                f"      Reasons: "
                f"{', '.join(frame['reason_codes'])}"
            )

    report_path = (
        "mock_data/outputs/keyframe_qa_report.json"
    )

    os.makedirs(
        os.path.dirname(report_path),
        exist_ok=True
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print(
        f"\nReport saved to: "
        f"{report_path}"
    )


if __name__ == "__main__":
    main()