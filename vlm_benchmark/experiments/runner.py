"""Experiment runner."""

from __future__ import annotations

import time
from collections import Counter

import pandas as pd
from tqdm import tqdm

from ..eval import answers_match, classify_error, score_mismatch
from ..models import run_inference
from ..types import BenchmarkSample


def run_experiment(
    cfg: dict,
    processor,
    model,
    samples: list[BenchmarkSample],
) -> pd.DataFrame:
    dataset_type = cfg.get("dataset_type", "gsm8k")
    records = []
    mode = cfg["experiment_mode"]
    n = len(samples)

    for i in tqdm(range(n), desc=f"[{mode}|{dataset_type}]"):
        if mode == "mismatch":
            txt_idx = (i + 1) % n
            q_text = samples[txt_idx].question
            img = samples[i].image
            img_index = i
        else:
            q_text = samples[i].question if mode != "image_only" else None
            img = samples[i].image if mode != "text_only" else None
            img_index = i

        ref = samples[i].answer
        t0 = time.time()
        pred = run_inference(processor, model, q_text, img, cfg)
        elapsed = time.time() - t0

        row = {
            "problem_id": i,
            "image_index": img_index,
            "question_text": q_text or "",
            "ground_truth": ref,
            "prediction": pred,
            "correct": answers_match(pred, ref, dataset_type),
            "error_type": classify_error(pred, ref, dataset_type),
            "elapsed_s": round(elapsed, 2),
        }

        if mode == "mismatch":
            mm = score_mismatch(pred, samples[i].answer, samples[txt_idx].answer, dataset_type)
            row.update(
                {
                    "mismatch_follows": mm["follows"],
                    "mismatch_img_diff": mm["img_diff"],
                    "mismatch_txt_diff": mm["txt_diff"],
                }
            )

        records.append(row)

    return pd.DataFrame(records)


def summarize_results(cfg: dict, df: pd.DataFrame) -> dict:
    accuracy = df["correct"].mean() if len(df) else 0.0
    error_counts = Counter(df["error_type"])
    follow_counts = Counter(df["mismatch_follows"]) if "mismatch_follows" in df.columns else Counter()

    return {
        "run_name": cfg["run_name"],
        "model_id": cfg["model_id"],
        "dataset_type": cfg.get("dataset_type", "gsm8k"),
        "experiment_mode": cfg["experiment_mode"],
        "num_problems": len(df),
        "accuracy": round(float(accuracy), 4),
        "n_correct": int(df["correct"].sum()),
        "error_counts": dict(error_counts),
        "follow_counts": dict(follow_counts),
        "avg_s_per_sample": round(float(df["elapsed_s"].mean()), 2),
        "total_minutes": round(float(df["elapsed_s"].sum() / 60), 1),
    }

