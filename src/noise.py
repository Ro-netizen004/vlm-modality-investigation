"""
Visual noise pipeline for ablation studies.

Applies controlled degradations to rendered text images to study
how VLM performance degrades as visual quality decreases.

Noise levels are organized as a gradient from clean to heavily degraded:

  Level 0: Clean (baseline — original PIL rendering)
  Level 1: Mild JPEG compression (quality=50)
  Level 2: Light Gaussian blur (radius=1)
  Level 3: Moderate JPEG compression (quality=20)
  Level 4: Medium blur (radius=2) + mild noise
  Level 5: Heavy blur (radius=3) + strong noise
  Level 6: Rotation (±5°) + perspective skew
  Level 7: Handwriting-style font + paper texture
  Level 8: Screenshot simulation (browser chrome, anti-aliasing)
  Level 9: Combined worst-case (blur + noise + rotation + compression)

Each noise function takes a PIL Image and returns a degraded PIL Image.
All operations are deterministic given a seed for reproducibility.
"""

import io
import math
import random
import textwrap
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance


# ══════════════════════════════════════════════════════════════════════════════
#  INDIVIDUAL NOISE TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def jpeg_compress(img: Image.Image, quality: int = 50) -> Image.Image:
    """Apply JPEG compression artifacts."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def gaussian_blur(img: Image.Image, radius: float = 1.0) -> Image.Image:
    """Apply Gaussian blur."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def add_gaussian_noise(img: Image.Image, std: float = 15.0, seed: int = 42) -> Image.Image:
    """Add Gaussian noise to image."""
    rng = np.random.RandomState(seed)
    arr = np.array(img, dtype=np.float32)
    noise = rng.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def salt_pepper_noise(img: Image.Image, prob: float = 0.02, seed: int = 42) -> Image.Image:
    """Add salt-and-pepper noise."""
    rng = np.random.RandomState(seed)
    arr = np.array(img)
    mask = rng.random(arr.shape[:2])
    arr[mask < prob / 2] = 0        # pepper
    arr[mask > 1 - prob / 2] = 255  # salt
    return Image.fromarray(arr)


def rotate_image(img: Image.Image, angle: float = 3.0) -> Image.Image:
    """Rotate image by specified degrees with white background fill."""
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))


def perspective_skew(img: Image.Image, magnitude: float = 0.05, seed: int = 42) -> Image.Image:
    """Apply mild perspective distortion."""
    rng = np.random.RandomState(seed)
    w, h = img.size
    # Random perspective offsets
    dx = int(w * magnitude)
    dy = int(h * magnitude)

    coeffs = [
        rng.randint(-dx, dx), rng.randint(-dy, dy),  # top-left
        rng.randint(-dx, dx), rng.randint(-dy, dy),  # top-right
        rng.randint(-dx, dx), rng.randint(-dy, dy),  # bottom-right
        rng.randint(-dx, dx), rng.randint(-dy, dy),  # bottom-left
    ]

    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(src[i][0] + coeffs[i * 2], src[i][1] + coeffs[i * 2 + 1]) for i in range(4)]

    # Use affine transform as approximation (perspective needs 8 coefficients)
    # Simple approach: just use the rotation + slight scaling
    return img.transform(img.size, Image.AFFINE,
                         (1 + magnitude * (rng.random() - 0.5),
                          magnitude * (rng.random() - 0.5), 0,
                          magnitude * (rng.random() - 0.5),
                          1 + magnitude * (rng.random() - 0.5), 0),
                         resample=Image.BICUBIC,
                         fillcolor=(255, 255, 255))


def adjust_contrast(img: Image.Image, factor: float = 0.7) -> Image.Image:
    """Reduce contrast."""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def adjust_brightness(img: Image.Image, factor: float = 0.85) -> Image.Image:
    """Reduce brightness (simulate dim scan)."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def add_paper_texture(img: Image.Image, intensity: float = 20.0, seed: int = 42) -> Image.Image:
    """Simulate paper texture / yellowing."""
    rng = np.random.RandomState(seed)
    arr = np.array(img, dtype=np.float32)
    # Slight yellow tint
    arr[:, :, 0] = np.clip(arr[:, :, 0] + 8, 0, 255)   # R
    arr[:, :, 1] = np.clip(arr[:, :, 1] + 5, 0, 255)   # G
    arr[:, :, 2] = np.clip(arr[:, :, 2] - 3, 0, 255)   # B
    # Fine grain noise
    grain = rng.normal(0, intensity * 0.3, arr.shape)
    arr = np.clip(arr + grain, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def downscale_upscale(img: Image.Image, factor: float = 0.5) -> Image.Image:
    """Simulate low-resolution capture by downscaling then upscaling."""
    w, h = img.size
    small = img.resize((int(w * factor), int(h * factor)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def add_shadow(img: Image.Image, seed: int = 42) -> Image.Image:
    """Add a simulated shadow/lighting gradient across the image."""
    rng = np.random.RandomState(seed)
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Random gradient direction
    direction = rng.choice(["left", "right", "top", "bottom"])
    if direction == "left":
        gradient = np.linspace(0.7, 1.0, w)[np.newaxis, :, np.newaxis]
    elif direction == "right":
        gradient = np.linspace(1.0, 0.7, w)[np.newaxis, :, np.newaxis]
    elif direction == "top":
        gradient = np.linspace(0.7, 1.0, h)[:, np.newaxis, np.newaxis]
    else:
        gradient = np.linspace(1.0, 0.7, h)[:, np.newaxis, np.newaxis]

    arr = np.clip(arr * gradient, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ══════════════════════════════════════════════════════════════════════════════
#  HANDWRITING-STYLE RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def render_handwriting_style(text: str, width: int = 900, seed: int = 42) -> Image.Image:
    """
    Render text in a handwriting-like style with imperfections.
    Uses available monospace/serif fonts with random jitter.
    """
    rng = np.random.RandomState(seed)
    font_size = 22
    padding = 40

    # Try different fonts for variety
    font_candidates = [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/georgia.ttf",
    ]
    font = ImageFont.load_default()
    for fp in font_candidates:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except OSError:
            continue

    chars_per_line = max(20, int((width - 2 * padding) / (font_size * 0.6)))
    wrapped = textwrap.fill(text, width=chars_per_line)
    lines = wrapped.split("\n")
    line_height = font_size + 10
    height = 2 * padding + len(lines) * line_height + 20

    # Cream/paper background
    bg = (252, 248, 235)
    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        x = padding + rng.randint(-2, 3)  # slight horizontal jitter
        y_jitter = rng.randint(-1, 2)
        # Vary ink color slightly
        ink = (rng.randint(10, 50), rng.randint(10, 50), rng.randint(10, 50))
        draw.text((x, y + y_jitter), line, fill=ink, font=font)
        y += line_height + rng.randint(-1, 2)

    # Add paper texture
    img = add_paper_texture(img, intensity=15.0, seed=seed)
    return img


# ══════════════════════════════════════════════════════════════════════════════
#  SCREENSHOT SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def simulate_screenshot(img: Image.Image, seed: int = 42) -> Image.Image:
    """
    Simulate a browser screenshot: add browser chrome, toolbar,
    padding, and anti-aliasing artifacts.
    """
    rng = np.random.RandomState(seed)
    w, h = img.size

    # Browser chrome dimensions
    toolbar_h = 60
    border = 2
    total_w = w + 2 * border + 20  # side margins
    total_h = h + toolbar_h + border + 20

    screenshot = Image.new("RGB", (total_w, total_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(screenshot)

    # Toolbar background
    draw.rectangle([(0, 0), (total_w, toolbar_h)], fill=(222, 222, 222))
    # Fake URL bar
    draw.rectangle([(80, 15), (total_w - 80, 45)], fill=(255, 255, 255),
                   outline=(200, 200, 200))
    # Fake buttons (three dots)
    for bx in [20, 35, 50]:
        draw.ellipse([(bx, 22), (bx + 12, 34)],
                     fill=[(255, 95, 87), (255, 189, 46), (39, 201, 63)][
                         (bx - 20) // 15])

    # Paste content
    screenshot.paste(img, (10, toolbar_h + 10))

    # Add slight anti-aliasing via mild blur
    screenshot = screenshot.filter(ImageFilter.GaussianBlur(radius=0.3))

    return screenshot


# ══════════════════════════════════════════════════════════════════════════════
#  NOISE LEVELS (ordered gradient from clean to worst-case)
# ══════════════════════════════════════════════════════════════════════════════

NOISE_LEVELS = {
    0: {
        "name": "clean",
        "description": "Original PIL rendering (baseline)",
        "transforms": [],
    },
    1: {
        "name": "jpeg_mild",
        "description": "Mild JPEG compression (quality=50)",
        "transforms": [("jpeg_compress", {"quality": 50})],
    },
    2: {
        "name": "blur_light",
        "description": "Light Gaussian blur (radius=1)",
        "transforms": [("gaussian_blur", {"radius": 1.0})],
    },
    3: {
        "name": "jpeg_heavy",
        "description": "Heavy JPEG compression (quality=15)",
        "transforms": [("jpeg_compress", {"quality": 15})],
    },
    4: {
        "name": "blur_noise",
        "description": "Medium blur (r=2) + Gaussian noise (std=15)",
        "transforms": [
            ("gaussian_blur", {"radius": 2.0}),
            ("add_gaussian_noise", {"std": 15.0}),
        ],
    },
    5: {
        "name": "heavy_degradation",
        "description": "Heavy blur (r=3) + strong noise (std=25) + contrast loss",
        "transforms": [
            ("gaussian_blur", {"radius": 3.0}),
            ("add_gaussian_noise", {"std": 25.0}),
            ("adjust_contrast", {"factor": 0.7}),
        ],
    },
    6: {
        "name": "geometric",
        "description": "Rotation (3°) + perspective skew + shadow",
        "transforms": [
            ("rotate_image", {"angle": 3.0}),
            ("perspective_skew", {"magnitude": 0.03}),
            ("add_shadow", {}),
        ],
    },
    7: {
        "name": "handwriting",
        "description": "Handwriting-style font + paper texture",
        "transforms": [("__handwriting__", {})],  # special case: re-renders
    },
    8: {
        "name": "screenshot",
        "description": "Browser screenshot simulation",
        "transforms": [("simulate_screenshot", {})],
    },
    9: {
        "name": "worst_case",
        "description": "Combined: blur + noise + rotation + JPEG(20) + low contrast",
        "transforms": [
            ("gaussian_blur", {"radius": 2.0}),
            ("add_gaussian_noise", {"std": 20.0}),
            ("rotate_image", {"angle": 4.0}),
            ("jpeg_compress", {"quality": 20}),
            ("adjust_contrast", {"factor": 0.75}),
            ("adjust_brightness", {"factor": 0.9}),
        ],
    },
}

# Map transform names to functions
TRANSFORM_REGISTRY = {
    "jpeg_compress": jpeg_compress,
    "gaussian_blur": gaussian_blur,
    "add_gaussian_noise": add_gaussian_noise,
    "salt_pepper_noise": salt_pepper_noise,
    "rotate_image": rotate_image,
    "perspective_skew": perspective_skew,
    "adjust_contrast": adjust_contrast,
    "adjust_brightness": adjust_brightness,
    "add_paper_texture": add_paper_texture,
    "downscale_upscale": downscale_upscale,
    "add_shadow": add_shadow,
    "simulate_screenshot": simulate_screenshot,
}


def apply_noise_level(img: Image.Image, level: int, text: str = None,
                      seed: int = 42) -> Image.Image:
    """
    Apply a specific noise level to an image.

    Args:
        img: Clean rendered image
        level: Noise level (0-9)
        text: Original text (needed for handwriting re-render at level 7)
        seed: Random seed for reproducibility
    """
    if level not in NOISE_LEVELS:
        raise ValueError(f"Unknown noise level: {level}. Valid: 0-9")

    config = NOISE_LEVELS[level]

    if level == 0:
        return img  # no-op

    for transform_name, kwargs in config["transforms"]:
        if transform_name == "__handwriting__":
            if text is None:
                raise ValueError("Level 7 (handwriting) requires the original text")
            return render_handwriting_style(text, width=img.size[0], seed=seed)

        transform_fn = TRANSFORM_REGISTRY[transform_name]
        # Add seed to kwargs if the function accepts it
        if "seed" in transform_fn.__code__.co_varnames:
            kwargs = {**kwargs, "seed": seed}
        img = transform_fn(img, **kwargs)

    return img


def apply_all_noise_levels(img: Image.Image, text: str = None,
                           seed: int = 42) -> dict:
    """Apply all noise levels and return dict of {level: degraded_image}."""
    results = {}
    for level in NOISE_LEVELS:
        results[level] = apply_noise_level(img, level, text=text, seed=seed)
    return results


def get_noise_level_names() -> dict:
    """Return {level: name} mapping."""
    return {level: config["name"] for level, config in NOISE_LEVELS.items()}


def get_noise_level_descriptions() -> dict:
    """Return {level: description} mapping."""
    return {level: config["description"] for level, config in NOISE_LEVELS.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH RENDERING WITH NOISE
# ══════════════════════════════════════════════════════════════════════════════

def render_noisy_images(questions: list, base_image_dir: str,
                        noise_levels: list = None, seed: int = 42):
    """
    Render images at multiple noise levels, saving to organized directories.

    Directory structure:
        base_image_dir/
            level_0_clean/q000.png, q001.png, ...
            level_1_jpeg_mild/q000.png, q001.png, ...
            ...
    """
    import os
    from tqdm import tqdm
    from src.rendering import render_text_to_image

    if noise_levels is None:
        noise_levels = list(NOISE_LEVELS.keys())

    for level in noise_levels:
        config = NOISE_LEVELS[level]
        level_dir = os.path.join(base_image_dir, f"level_{level}_{config['name']}")
        os.makedirs(level_dir, exist_ok=True)

        # Check if already done
        existing = sum(1 for f in os.listdir(level_dir) if f.endswith(".png"))
        if existing >= len(questions):
            print(f"  Level {level} ({config['name']}): {existing} images exist, skipping")
            continue

        print(f"  Rendering level {level} ({config['name']}): {config['description']}")
        for i in tqdm(range(len(questions)), desc=f"Level {level}"):
            out_path = os.path.join(level_dir, f"q{i:03d}.png")
            if os.path.exists(out_path):
                continue

            # Start from clean render
            clean = render_text_to_image(questions[i])
            noisy = apply_noise_level(clean, level, text=questions[i], seed=seed + i)
            noisy.save(out_path)
