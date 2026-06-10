"""Image rendering utilities for converting text problems to PNG images."""

import os
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
        img = render_text_to_image(
            questions[i],
            width=config.get("width", 900),
            font_size=config.get("font_size", 22),
            padding=config.get("padding", 40),
            bg_color=config.get("bg_color", "white"),
            text_color=config.get("text_color", "black"),
        )
        img.save(os.path.join(image_dir, f"q{i:03d}.png"))
    print("Done rendering.")
