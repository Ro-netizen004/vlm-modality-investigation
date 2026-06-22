"""Image rendering utilities for converting text problems to PNG images."""

import io
import os
import re
import textwrap
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def render_text_to_image(
    text: str,
    width: int = 900,
    font_size: int = 22,
    padding: int = 40,
    bg_color: str = "white",
    text_color: str = "black",
) -> Image.Image:
    """Render a math problem string into a clean PNG image."""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

    chars_per_line = max(20, int((width - 2 * padding) / (font_size * 0.6)))
    wrapped = textwrap.fill(text, width=chars_per_line)
    lines = wrapped.split("\n")
    line_height = font_size + 8
    height = 2 * padding + len(lines) * line_height + 10

    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=text_color, font=font)
        y += line_height
    return img


def _has_latex(text: str) -> bool:
    """Return True if text contains LaTeX math expressions."""
    return bool(re.search(r"\$[^$]+\$|\\frac|\\sqrt|\\sum|\\int|\\theta|\\pi|\\le|\\ge|\\boxed", text))


def render_latex_to_image(
    text: str,
    width: int = 900,
    font_size: int = 14,
    padding: int = 40,
    bg_color: str = "white",
    text_color: str = "black",
) -> Image.Image:
    """
    Render a math problem containing LaTeX into a PNG using matplotlib.
    Renders each line as a separate matplotlib text object so mathtext
    interprets $...$ expressions as formatted math symbols.
    Falls back to render_text_to_image if matplotlib fails.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
    except ImportError:
        return render_text_to_image(text, width, font_size + 8, padding, bg_color, text_color)

    try:
        dpi = 100
        fig_width = width / dpi
        chars_per_line = max(40, int((width - 2 * padding) / (font_size * 0.6)))

        # Word-wrap plain segments, keep $...$ intact
        segments = re.split(r"(\$[^$]+\$)", text)
        rebuilt = []
        for seg in segments:
            if seg.startswith("$") and seg.endswith("$"):
                rebuilt.append(seg)
            else:
                rebuilt.append(textwrap.fill(seg, width=chars_per_line))
        wrapped_text = "".join(rebuilt)
        lines = wrapped_text.split("\n")

        line_height_in = (font_size + 6) / dpi
        fig_height = max(1.5, len(lines) * line_height_in + 2 * padding / dpi)

        fig = Figure(figsize=(fig_width, fig_height), dpi=dpi)
        fig.patch.set_facecolor(bg_color)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(0, width)
        ax.set_ylim(0, fig_height * dpi)
        ax.set_facecolor(bg_color)

        y = fig_height * dpi - padding
        for line in lines:
            ax.text(
                padding, y, line,
                fontsize=font_size,
                color=text_color,
                verticalalignment="top",
                usetex=False,  # matplotlib mathtext — no LaTeX install needed
            )
            y -= line_height_in * dpi

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=bg_color,
                    bbox_inches=None)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        plt.close("all")
        return img

    except Exception:
        return render_text_to_image(text, width, font_size + 8, padding, bg_color, text_color)


def render_problem_to_image(
    text: str,
    width: int = 900,
    font_size: int = 22,
    padding: int = 40,
    bg_color: str = "white",
    text_color: str = "black",
    latex: bool = False,
) -> Image.Image:
    """
    Unified entry point — auto-detects LaTeX or uses explicit latex=True flag.
    Uses matplotlib renderer for LaTeX, PIL renderer for plain text.
    """
    if latex or _has_latex(text):
        return render_latex_to_image(text, width, font_size - 8, padding, bg_color, text_color)
    return render_text_to_image(text, width, font_size, padding, bg_color, text_color)


def load_image(i: int, image_dir: str) -> Image.Image:
    """Load a pre-rendered problem image from disk."""
    path = os.path.join(image_dir, f"q{i:03d}.png")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def render_all_images(questions, image_dir, render_config=None, force=False):
    """Pre-render all problem images, skipping existing files unless force=True."""
    os.makedirs(image_dir, exist_ok=True)
    config = render_config or {}

    if force:
        missing = list(range(len(questions)))
    else:
        missing = [
            i for i in range(len(questions))
            if not os.path.exists(os.path.join(image_dir, f"q{i:03d}.png"))
        ]

    if not missing:
        print(f"All {len(questions)} images already exist in {image_dir}")
        return

    print(f"Rendering {len(missing)} images...")
    for i in tqdm(missing, desc="Rendering"):
        img = render_problem_to_image(
            questions[i],
            width=config.get("width", 900),
            font_size=config.get("font_size", 22),
            padding=config.get("padding", 40),
            bg_color=config.get("bg_color", "white"),
            text_color=config.get("text_color", "black"),
        )
        img.save(os.path.join(image_dir, f"q{i:03d}.png"))
    print("Done rendering.")
