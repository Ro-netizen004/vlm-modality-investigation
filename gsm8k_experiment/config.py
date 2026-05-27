"""Experiment configuration helpers."""

from __future__ import annotations

from pathlib import Path

DEFAULT_CONFIG = {
  "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
  "use_4bit": True,
  # Logical dataset key → adapter + answer parser (gsm8k | svamp | …)
  "dataset_type": "gsm8k",
  "hf_dataset_id": "RodelaG/gsm8k-rendered-vlm",
  "dataset_split": "test",
  "metadata_csv": "data/gsm8k_metadata_clean.csv",
  "image_root": ".",
  "num_problems": 100,
  "experiment_mode": "text_and_image",
  "max_new_tokens": 512,
  "do_sample": False,
  "output_dir": "results",
  "run_name": None,
  "seed": 42,
}


def finalize_config(cfg: dict) -> dict:
    """Return a copy of cfg with derived output paths and directories created."""
    cfg = dict(cfg)

    if cfg.get("run_name") is None:
        model_tag = cfg["model_id"].split("/")[-1]
        dtype = cfg.get("dataset_type", "gsm8k")
        cfg["run_name"] = f"{model_tag}__{dtype}__{cfg['experiment_mode']}"

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    cfg["results_csv"] = str(out / f"{cfg['run_name']}_results.csv")
    cfg["summary_json"] = str(out / f"{cfg['run_name']}_summary.json")
    return cfg
