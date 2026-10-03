import gradio as gr

from src.qa_engine import QAEngine


engine = None


def run_qa(reference, generated):
    global engine

    if reference is None or generated is None:
        return (
            "REVIEW",
            "Please provide both reference and generated images.",
            "{}"
        )

    if engine is None:
        engine = QAEngine()

    reference_path = reference
    generated_path = generated

    result = engine.analyze(
        reference_path,
        generated_path
    )

    decision = result["decision"]

    reason_codes = result["reason_codes"]

    reason_text = (
        "\n".join(reason_codes)
        if reason_codes
        else "No failure reason codes"
    )

    import json

    report = json.dumps(
        result,
        indent=2
    )

    return (
        decision,
        reason_text,
        report
    )


with gr.Blocks(title="Identity & Visual Consistency QA") as demo:

    gr.Markdown(
        """
        # Identity & Visual Consistency QA

        Compare a reference image with a generated image.

        The system evaluates:

        - AdaFace identity similarity
        - SSIM structural similarity
        - LPIPS perceptual similarity
        - CLIP semantic similarity
        - Reason codes
        """
    )

    with gr.Row():

        reference_image = gr.Image(
            type="filepath",
            label="Reference Image"
        )

        generated_image = gr.Image(
            type="filepath",
            label="Generated Image"
        )

    analyze_button = gr.Button(
        "Run QA"
    )

    decision = gr.Textbox(
        label="Decision"
    )

    reasons = gr.Textbox(
        label="Reason Codes",
        lines=5
    )

    report = gr.Code(
        label="QA Report JSON",
        language="json",
        lines=20
    )

    analyze_button.click(
        fn=run_qa,
        inputs=[
            reference_image,
            generated_image
        ],
        outputs=[
            decision,
            reasons,
            report
        ]
    )


if __name__ == "__main__":
    demo.launch()