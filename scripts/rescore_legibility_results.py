"""Rescore completed legibility CSVs without loading model dependencies.

Usage:
    python scripts/rescore_legibility_results.py MODEL_DIR MODEL_KEY [--channel text]
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text_noise import TEXT_NOISE_LEVELS, degrade_text


REASONING_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "need", "to", "of", "in", "on", "at", "by", "for",
    "with", "about", "into", "from", "up", "out", "as", "if", "or", "and", "but",
    "not", "no", "so", "it", "its", "he", "she", "they", "we", "you", "i", "his",
    "her", "their", "our", "how", "what", "many", "much", "more", "each", "every",
    "per", "total", "find", "make", "take", "get", "give", "long", "old", "new",
    "all", "any", "some", "than",
}


def score_by_reasoning(prediction: str, image_question: str, text_question: str) -> str:
    """Dependency-light copy of the canonical reasoning-trace classifier."""
    def keywords(question: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z]+", str(question).lower())
        return {token for token in tokens if len(token) > 3 and token not in REASONING_STOP}

    def numbers(value: str) -> set[str]:
        return set(re.findall(r"\b\d+(?:\.\d+)?\b", str(value)))

    pred_lower = str(prediction).lower()
    pred_numbers = numbers(prediction)
    image_keywords, text_keywords = keywords(image_question), keywords(text_question)
    image_score = (
        sum(word in pred_lower for word in image_keywords - text_keywords) * 2
        + len(pred_numbers & (numbers(image_question) - numbers(text_question)))
    )
    text_score = (
        sum(word in pred_lower for word in text_keywords - image_keywords) * 2
        + len(pred_numbers & (numbers(text_question) - numbers(image_question)))
    )
    if text_score > image_score and text_score > 0:
        return "text_reasoning"
    if image_score > text_score and image_score > 0:
        return "image_reasoning"
    return "neither"


def write_json(path: Path, obj: dict) -> None:
    temp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)
    os.replace(temp, path)


def rescore_csv(path: Path, channel: str, level: int) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    raw, rescored = Counter(), Counter()
    n = len(rows)
    corrupt_text = (
        channel == "text" and TEXT_NOISE_LEVELS.get(level, {}).get("p", 0) > 0
    )
    for row in rows:
        follows = row["follows"]
        raw[follows] += 1
        if follows != "neither":
            rescored[follows] += 1
            continue

        text_question = row["text_question"]
        if corrupt_text:
            seed = (int(row["problem_id"]) + 1) % n
            text_question = degrade_text(text_question, level, seed=seed)
        rescored[
            score_by_reasoning(
                row["prediction"], row["image_question"], text_question
            )
        ] += 1

    raw_image, raw_text = raw.get("image", 0), raw.get("text", 0)
    raw_decidable = raw_image + raw_text
    image = rescored.get("image", 0) + rescored.get("image_reasoning", 0)
    text = rescored.get("text", 0) + rescored.get("text_reasoning", 0)
    decidable = image + text
    return {
        "n_problems": n,
        "counts_raw": dict(raw),
        "decidable_raw": raw_decidable,
        "text_preference_raw": raw_text / raw_decidable if raw_decidable else None,
        "counts_rescored": dict(rescored),
        "decidable": decidable,
        "text_preference": text / decidable if decidable else None,
        "neither_rate": rescored.get("neither", 0) / n if n else None,
        "rescore_channel": channel,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("model_key")
    parser.add_argument("--channel", choices=("image", "text"), default="image")
    parser.add_argument("--levels", nargs="+", type=int,
                        help="only rescore these levels")
    args = parser.parse_args()

    levels = {}
    for csv_path in sorted(args.model_dir.glob("level_*.csv")):
        match = re.fullmatch(r"level_(\d+)_(.+)\.csv", csv_path.name)
        if not match:
            continue
        level, name = int(match.group(1)), match.group(2)
        if args.levels is not None and level not in args.levels:
            continue
        result = rescore_csv(csv_path, args.channel, level)
        result.update(level=level, name=name)
        levels[level] = result
        write_json(args.model_dir / f"level_{level}_{name}_rescored.json", result)

    ordered = sorted(levels)
    summary = {
        "model": args.model_key,
        "text_preference_by_level": {
            level: levels[level]["text_preference"] for level in ordered
        },
        "text_preference_raw_by_level": {
            level: levels[level]["text_preference_raw"] for level in ordered
        },
        "decidable_by_level": {
            level: levels[level]["decidable"] for level in ordered
        },
        "decidable_raw_by_level": {
            level: levels[level]["decidable_raw"] for level in ordered
        },
    }
    write_json(args.model_dir / "legibility_summary_rescored.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
