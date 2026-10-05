import gradio as gr
import json
from pathlib import Path

from src.qa_engine import QAEngine


PROJECT_ROOT = Path(__file__).resolve().parent.parent


engine = QAEngine()


def run_qa(reference_image, generated_image):
    """
    Run the complete identity + visual QA pipeline
    and return a structured report for the UI.
    """

    if reference_image is None:
        return (
            "ERROR",
            {},
            "Please provide a reference image."
        )

    if generated_image is None:
        return (
            "ERROR",
            {},
            "Please provide a generated image."
        )

    try:
        result = engine.analyze(
            reference_image,
            generated_image
        )

        decision = result["decision"]

        if decision == "PASS":
            status = "PASS"
        else:
            status = "FAIL"

        return (
            status,
            result,
            json.dumps(
                result,
                indent=2
            )
        )

    except Exception as error:

        return (
            "ERROR",
            {},
            f"QA execution failed:\n{error}"
        )


def build_ui():

    with gr.Blocks(
        title="Identity & Visual Consistency QA"
    ) as demo:

        gr.Markdown(
            """
# Identity & Visual Consistency QA

Compare a reference image against an AI-generated image
using multiple independent QA signals.

### QA Components

- **AdaFace** — primary face identity verification
- **PixelFacePrototype** — prototype identity evidence
- **DINOv3** — visual representation similarity
- **SSIM** — structural similarity
- **LPIPS** — perceptual similarity
- **CLIP** — semantic similarity
- **Object consistency** — object-region consistency
- **Environment consistency** — background/environment consistency
- **Reason codes** — explains why a sample passed or failed
"""
        )

        with gr.Row():

            with gr.Column():

                reference_input = gr.Image(
                    label="Reference Image",
                    type="filepath"
                )

                generated_input = gr.Image(
                    label="Generated Image",
                    type="filepath"
                )

                run_button = gr.Button(
                    "Run QA",
                    variant="primary"
                )

            with gr.Column():

                decision_output = gr.Textbox(
                    label="QA Decision",
                    interactive=False
                )

                report_output = gr.JSON(
                    label="QA Report"
                )

        json_output = gr.Code(
            label="Raw QA Report JSON",
            language="json",
            interactive=False
        )

        gr.Markdown(
            """
## Interpretation

**PASS** means no configured QA failure reason was detected.

**FAIL** means at least one QA signal detected a configured
identity or visual consistency problem.

The system intentionally retains component-level evidence
instead of reducing the result to a single opaque score.
"""
        )

        run_button.click(
            fn=run_qa,
            inputs=[
                reference_input,
                generated_input
            ],
            outputs=[
                decision_output,
                report_output,
                json_output
            ]
        )

    return demo


if __name__ == "__main__":

    demo = build_ui()

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860
    )