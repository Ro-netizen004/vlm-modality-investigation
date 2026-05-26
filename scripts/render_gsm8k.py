import os
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import json
import subprocess
from tqdm import tqdm

# -------------------
# Load dataset
# -------------------
# Use namespace/name — required by recent huggingface_hub (bare "gsm8k" raises HfUriError).
dataset = load_dataset("openai/gsm8k", "main")["test"]
samples = list(dataset)  # materialize once for repeatable access / efficiency

# -------------------
# Output folder
# -------------------
OUTPUT_DIR = "rendered_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------
# Image settings
# -------------------
# Use a single shared render spec across models.
# 672px matches LLaVA-NeXT's native tile resolution and works well generally.
WIDTH = 672
PADDING = 40
FONT_SIZE = 22
LINE_SPACING = 10

# Prefer a repo-shipped font for strict reproducibility (if present).
# Put a TTF at: assets/fonts/DejaVuSans.ttf
REPO_FONT_PATH = os.path.join("assets", "fonts", "DejaVuSans.ttf")
REQUIRE_REPO_FONT = False  # set True for strict paper-grade determinism
font = None
font_used = "PIL_default"

if REQUIRE_REPO_FONT:
    assert os.path.exists(REPO_FONT_PATH), (
        f"Missing required font at {REPO_FONT_PATH}. "
        "Add the TTF to the repo or set REQUIRE_REPO_FONT=False."
    )

if os.path.exists(REPO_FONT_PATH):
    try:
        font = ImageFont.truetype(REPO_FONT_PATH, FONT_SIZE)
        font_used = REPO_FONT_PATH
    except Exception:
        font = None

# Fallback to common system fonts (less deterministic across environments).
if font is None:
    _FONT_CANDIDATES = [
        "DejaVuSans.ttf",                 # common on Linux/Colab
        "Arial.ttf", "arial.ttf",         # common on Windows/macOS
        "LiberationSans-Regular.ttf",     # common on Linux
    ]
    for name in _FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(name, FONT_SIZE)
            font_used = name
            break
        except Exception:
            pass

if font is None:
    font = ImageFont.load_default()

# -------------------
# Text wrapping
# -------------------
def wrap_text(text, font, max_width):
    words = text.split()
    lines = []
    current = ""

    for w in words:
        # If a single "word" is longer than the line, put it on its own line.
        # (Rare for GSM8K, but prevents infinite wrapping issues.)
        if font.getbbox(w)[2] > max_width:
            if current:
                lines.append(current.rstrip())
            lines.append(w)
            current = ""
            continue

        test = current + w + " "
        if font.getbbox(test)[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current.rstrip())
            current = w + " "

    if current:
        lines.append(current.rstrip())
    return lines

# -------------------
# Render loop
# -------------------
pad = 4  # deterministic for full GSM8K test (e.g., q0000.png ...)
SAMPLES_DIR = "render_samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)
METADATA_PATH = "gsm8k_metadata.csv"
PROGRESS_PATH = "render_progress.json"
CHECKPOINT_EVERY = 50  # rows

total = len(samples)
for i in tqdm(range(total)):
    sample = samples[i]
    text = "Solve this step-by-step:\n\n" + sample["question"]
    text = " ".join(text.split())

    lines = wrap_text(text, font, WIDTH - 2 * PADDING)

    line_height = FONT_SIZE + LINE_SPACING
    height = len(lines) * line_height + 2 * PADDING

    img = Image.new("RGB", (WIDTH, height), "white")
    draw = ImageDraw.Draw(img)

    y = PADDING
    for line in lines:
        draw.text((PADDING, y), line, fill="black", font=font)
        y += line_height

    filename = f"q{i:0{pad}d}.png"
    out_path = os.path.join(OUTPUT_DIR, filename)

    # Resume-safe: skip already-rendered images.
    if not os.path.exists(out_path):
        img.save(out_path)
        if i < 10:
            img.save(os.path.join(SAMPLES_DIR, filename))

    # Periodic checkpoint so long runs can resume safely.
    if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == total:
        with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
            json.dump({"rendered": i + 1, "total": total}, f, indent=2)

print("Done generating images.")

# -------------------
# Validation (basic pipeline sanity checks)
# -------------------
expected_files = {f"q{i:0{pad}d}.png" for i in range(total)}
present_files = {
    fn
    for fn in os.listdir(OUTPUT_DIR)
    if fn.lower().endswith(".png") and fn.startswith("q")
}
missing = sorted(expected_files - present_files)
extra = sorted(present_files - expected_files)
assert not missing, f"Missing {len(missing)} expected PNG(s) in {OUTPUT_DIR}. Example: {missing[:5]}"
assert not extra, f"Found {len(extra)} unexpected PNG(s) in {OUTPUT_DIR}. Example: {extra[:5]}"
print(f"Validated: {len(present_files)} PNGs exactly match expected set.")

# -------------------
# Metadata for cloud inference (deterministic rebuild)
# -------------------
df = pd.DataFrame(
    [
        {
            "id": i,
            "question": samples[i]["question"],
            "answer": samples[i]["answer"],
            "image": f"q{i:0{pad}d}.png",
        }
        for i in range(total)
    ]
)
df.to_csv(METADATA_PATH, index=False)
print(f"Saved metadata to {METADATA_PATH}")

# -------------------
# Reproducibility artifact (paper/methods ready)
# -------------------
dataset_fingerprint = getattr(dataset, "_fingerprint", None)
git_commit = None
try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    git_commit = None

render_config = {
    "dataset": {
        "name": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "size": len(samples),
        "fingerprint": dataset_fingerprint,
    },
    "image_format": "PNG",
    "naming_convention": f"q{{id:0{pad}d}}.png",
    "width_px": WIDTH,
    "height_px": "variable",
    "padding_px": PADDING,
    "font_size_px": FONT_SIZE,
    "line_spacing_px": LINE_SPACING,
    "font_used": font_used,
    "repo_font_path_preferred": REPO_FONT_PATH,
    "require_repo_font": REQUIRE_REPO_FONT,
    "background_color": "white",
    "text_color": "black",
    "wrapping": "greedy word-wrap; overlong single tokens placed on their own line",
    "prompt_header": "Solve this step-by-step:",
    "clean_rendering_only": True,
    "noise": {"enabled": False, "notes": "No blur, compression, rotation, or background noise applied."},
    "samples_dir": SAMPLES_DIR,
    "samples_count": 10,
    "checkpointing": {
        "enabled": True,
        "every_rows": CHECKPOINT_EVERY,
        "progress_file": PROGRESS_PATH,
        "note": "Images are resume-safe via file existence; metadata is rebuilt deterministically at end.",
    },
    "git_commit": git_commit,
}
with open("render_config.json", "w", encoding="utf-8") as f:
    json.dump(render_config, f, indent=2)
print("Saved render config to render_config.json")