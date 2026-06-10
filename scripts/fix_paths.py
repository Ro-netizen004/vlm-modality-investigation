import re
import os
import pandas as pd

DATA_DIR = "data"
INPUT_CANDIDATES = [
    os.path.join(DATA_DIR, "gsm8k_metadata.csv"),
    os.path.join(DATA_DIR, "gsm8k_metadata_fixed.csv"),
    os.path.join(DATA_DIR, "gsm8k_metadata_clean.csv"),
]
OUTPUT_PATH = os.path.join(DATA_DIR, "gsm8k_metadata_clean.csv")


def extract_answer(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    match = re.findall(r"####\s*([+-]?\d+(?:\.\d+)?)", text)
    return match[-1] if match else text.strip()


def extract_reasoning(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    reasoning = text.split("####")[0]
    # Remove GSM8K inline calculation artifacts like <<2+3=5>>.
    reasoning = re.sub(r"<<.*?>>", "", reasoning)
    # Normalize spacing/newlines after artifact removal.
    reasoning = re.sub(r"[ \t]+", " ", reasoning)
    reasoning = re.sub(r"\n\s*\n+", "\n", reasoning)
    return reasoning.strip()


def normalize_image_path(path: str) -> str:
    path = "" if pd.isna(path) else str(path).strip()
    if path.startswith("rendered_images/") or path.startswith("rendered_images\\"):
        return path.replace("\\", "/")
    return f"rendered_images/{path}".replace("\\", "/")


INPUT_PATH = next((p for p in INPUT_CANDIDATES if os.path.exists(p)), None)
if INPUT_PATH is None:
    raise FileNotFoundError(
        "No metadata input found. Expected one of: "
        + ", ".join(INPUT_CANDIDATES)
    )

df = pd.read_csv(INPUT_PATH)

# Use the original answer text for both derived fields.
raw_answer = df["answer"].copy()
df["reasoning"] = raw_answer.apply(extract_reasoning)
df["answer"] = raw_answer.apply(extract_answer)

# Ensure image paths are consistently prefixed for cloud/HF loading.
df["image"] = df["image"].apply(normalize_image_path)

df.to_csv(OUTPUT_PATH, index=False)
print(f"Input : {os.path.abspath(INPUT_PATH)}")
print(f"Saved {len(df)} rows to: {os.path.abspath(OUTPUT_PATH)}")