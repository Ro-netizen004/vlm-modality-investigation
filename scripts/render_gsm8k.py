import os
import json
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from tqdm import tqdm

HF_DATASET_REPO = "RodelaG/gsm8k-rendered-vlm"

dataset = load_dataset("openai/gsm8k", "main")["test"]
samples = list(dataset)

OUTPUT_DIR = "rendered_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

WIDTH = 672
PADDING = 40
FONT_SIZE = 22
LINE_SPACING = 10

font = None
for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf", "LiberationSans-Regular.ttf"):
    try:
        font = ImageFont.truetype(name, FONT_SIZE)
        break
    except OSError:
        pass
if font is None:
    font = ImageFont.load_default()

METADATA_PATH = os.path.join(DATA_DIR, "gsm8k_metadata.csv")
pad = 4


def wrap_text(text, font, max_width):
    words = text.split()
    lines = []
    current = ""

    for w in words:
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


for i in tqdm(range(len(samples))):
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
    if not os.path.exists(out_path):
        img.save(out_path)

print("Done generating images.")

expected_files = {f"q{i:0{pad}d}.png" for i in range(len(samples))}
present_files = {
    fn for fn in os.listdir(OUTPUT_DIR) if fn.lower().endswith(".png") and fn.startswith("q")
}
missing = sorted(expected_files - present_files)
extra = sorted(present_files - expected_files)
assert not missing, f"Missing PNGs in {OUTPUT_DIR}. Example: {missing[:5]}"
assert not extra, f"Unexpected PNGs in {OUTPUT_DIR}. Example: {extra[:5]}"

pd.DataFrame(
    [
        {
            "id": i,
            "question": samples[i]["question"],
            "answer": samples[i]["answer"],
            "image": f"q{i:0{pad}d}.png",
        }
        for i in range(len(samples))
    ]
).to_csv(METADATA_PATH, index=False)
print(f"Saved metadata to {METADATA_PATH}")

render_config_path = os.path.join(DATA_DIR, "render_config.json")
with open(render_config_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "huggingface_dataset": HF_DATASET_REPO,
            "source": "openai/gsm8k",
            "split": "test",
            "width_px": WIDTH,
            "font_size_px": FONT_SIZE,
            "padding_px": PADDING,
            "naming": f"q{{id:0{pad}d}}.png",
        },
        f,
        indent=2,
    )
print(f"Saved render config to {render_config_path}")
