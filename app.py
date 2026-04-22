"""
Gradio app — Paint by Number Generator
Runs on Hugging Face Spaces.
"""

import io
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from paint_by_number import run, color_name, rgb_to_hex


def process(image, n_colors, scale, smooth):
    """Gradio handler — takes PIL image, returns (pbn PIL, palette PIL)."""
    if image is None:
        return None, None

    img = Image.fromarray(image).convert("RGB")

    pbn, pal_fig, palette = run(
        img,
        n_colors  = int(n_colors),
        scale     = int(scale),
        smooth    = int(smooth),
    )

    # Convert palette figure → PIL image
    buf = io.BytesIO()
    pal_fig.savefig(buf, format="png", dpi=120,
                    bbox_inches="tight", facecolor="#f8f8f8")
    plt.close(pal_fig)
    buf.seek(0)
    pal_img = Image.open(buf).copy()

    return pbn, pal_img


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Paint by Number Generator") as demo:
    gr.Markdown(
        """
        # 🎨 Paint by Number Generator
        Upload any photo and get a paint-by-number guide — great for kids!
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp_image  = gr.Image(label="Upload your photo", type="numpy")
            n_colors   = gr.Slider(4, 24, value=14, step=1,
                                   label="Number of colors")
            scale      = gr.Slider(4, 12, value=8,  step=1,
                                   label="Output scale (higher = larger image)")
            smooth     = gr.Slider(0, 3,  value=1,  step=1,
                                   label="Smoothing passes")
            btn        = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            out_pbn    = gr.Image(label="Paint by Number guide")
            out_palette = gr.Image(label="Color palette")

    btn.click(
        fn      = process,
        inputs  = [inp_image, n_colors, scale, smooth],
        outputs = [out_pbn, out_palette],
    )

    gr.Markdown(
        """
        ---
        **Tips:**
        - 14 colors works well for most photos
        - Increase scale for a larger printable output
        - Source code on [GitHub](https://github.com/YOUR_USERNAME/paint-by-number)
        """
    )


if __name__ == "__main__":
    demo.launch()
