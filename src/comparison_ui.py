import json
import tempfile
from pathlib import Path

import gradio as gr

from src.qa_engine import QAEngine


PROJECT_ROOT = Path(__file__).resolve().parent.parent

engine = QAEngine()


def run_video_qa(reference_image, video_clip, clip_metadata, sample_count):
    """
    Run the Day 2 video QA pipeline.

    Inputs:
        reference_image: reference identity image
        video_clip: generated MP4 clip
        clip_metadata: ClipMetadata JSON file
        sample_count: deterministic coarse sample count

    Outputs:
        decision, QA report, raw JSON
    """

    if reference_image is None:
        return (
            "ERROR",
            {},
            "Please provide a reference image.",
        )

    if video_clip is None:
        return (
            "ERROR",
            {},
            "Please provide a video clip.",
        )

    if clip_metadata is None:
        return (
            "ERROR",
            {},
            "Please provide the ClipMetadata JSON file.",
        )

    try:
        reference_image = Path(reference_image)
        video_clip = Path(video_clip)
        clip_metadata = Path(clip_metadata)

        if not reference_image.exists():
            raise FileNotFoundError(
                f"Reference image not found: {reference_image}"
            )

        if not video_clip.exists():
            raise FileNotFoundError(
                f"Video clip not found: {video_clip}"
            )

        if not clip_metadata.exists():
            raise FileNotFoundError(
                f"Clip metadata not found: {clip_metadata}"
            )

        # Create a dedicated output directory for this QA run.
        output_dir = Path(
            tempfile.mkdtemp(
                prefix="member7_video_qa_",
                dir=PROJECT_ROOT / "mock_data" / "outputs",
            )
        )

        # Run the existing Day 2 video QA implementation.
        result = engine.analyze_clip(
            reference_path=str(reference_image),
            video_path=str(video_clip),
            output_dir=str(output_dir),
            sample_count=int(sample_count),
            clip_metadata_path=str(clip_metadata),
        )

        decision = result.get("decision", "UNKNOWN")

        return (
            decision,
            result,
            json.dumps(result, indent=2),
        )

    except Exception as error:
        return (
            "ERROR",
            {},
            f"QA execution failed:\n{error}",
        )


def build_ui():
    with gr.Blocks(
        title="Identity & Visual Consistency QA"
    ) as demo:

        gr.Markdown(
            """
            # Identity & Visual Consistency QA

            Day 2 video QA POC for Member 7.

            The system compares a reference identity image against
            frames sampled from a generated video clip and produces
            a structured QAResult.

            **Pipeline:**

            `Video + ClipMetadata → Frame Sampling → Identity & Visual QA → Decision`
            """
        )

        gr.Markdown(
            """
            ## QA Components

            - **AdaFace** — primary face identity verification
            - **PixelFacePrototype** — prototype identity evidence
            - **DINOv3** — visual representation similarity
            - **SSIM** — structural similarity
            - **LPIPS** — perceptual similarity
            - **CLIP** — semantic similarity
            - **Object consistency** — object-region consistency
            - **Environment consistency** — background/environment consistency
            - **Reason codes** — explain detected failures
            """
        )

        with gr.Row():

            with gr.Column():

                reference_input = gr.Image(
                    label="Reference Image",
                    type="filepath",
                )

                video_input = gr.Video(
                    label="Generated Video Clip",
                    sources=["upload"],
                )

                metadata_input = gr.File(
                    label="Clip Metadata JSON",
                    file_types=[".json"],
                    type="filepath",
                )

                sample_count = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Coarse Sample Count",
                )

                run_button = gr.Button(
                    "Run Video QA",
                    variant="primary",
                )

            with gr.Column():

                decision_output = gr.Textbox(
                    label="QA Decision",
                    interactive=False,
                )

                report_output = gr.JSON(
                    label="QA Report",
                )

        json_output = gr.Code(
            label="Raw QAResult JSON",
            language="json",
            interactive=False,
        )

        gr.Markdown(
            """
            ## Decision Interpretation

            **PASS**  
            No configured QA failure reason was detected.

            **AUTO_RETRY**  
            A high-confidence identity or severe visual failure was detected.

            **HUMAN_REVIEW**  
            A visual change was detected that requires human review.

            **ERROR**  
            The QA pipeline could not execute because of an input or runtime error.
            """
        )

        run_button.click(
            fn=run_video_qa,
            inputs=[
                reference_input,
                video_input,
                metadata_input,
                sample_count,
            ],
            outputs=[
                decision_output,
                report_output,
                json_output,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
    )