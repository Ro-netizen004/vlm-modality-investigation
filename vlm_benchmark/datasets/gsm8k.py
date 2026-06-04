"""GSM8K rendered-VL dataset adapter."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..types import BenchmarkSample
from .base import register_adapter


class GSM8KAdapter:
    dataset_type = "gsm8k"

    def load(self, cfg: dict) -> list[BenchmarkSample]:
        metadata_csv = cfg.get("metadata_csv")
        image_root = Path(cfg.get("image_root", "."))

        if metadata_csv and Path(metadata_csv).exists():
            return self._from_metadata_csv(metadata_csv, image_root, cfg.get("num_problems"))

        hf_id = cfg.get("hf_dataset_id") or cfg.get("dataset_name")
        if hf_id and "/" in str(hf_id):
            return self._from_hf(hf_id, cfg)

        raise FileNotFoundError(
            "GSM8K: provide metadata_csv + image_root, or hf_dataset_id (HF repo with /)."
        )

    def _from_metadata_csv(
        self, metadata_csv: str, image_root: Path, num_problems: int | None
    ) -> list[BenchmarkSample]:
        import pandas as pd

        print(f"GSM8K: loading metadata {metadata_csv}")
        df = pd.read_csv(metadata_csv).sort_values("id", kind="stable").reset_index(drop=True)
        if num_problems is not None:
            df = df.head(num_problems)

        samples: list[BenchmarkSample] = []
        for _, row in df.iterrows():
            img = None
            if pd.notna(row.get("image")):
                path = image_root / str(row["image"])
                if path.exists():
                    img = Image.open(path).convert("RGB")

            meta = {"id": row.get("id"), "reasoning": row.get("reasoning")}
            meta = {k: v for k, v in meta.items() if pd.notna(v)}
            samples.append(
                BenchmarkSample(
                    question=str(row["question"]),
                    answer=str(row["answer"]),
                    image=img,
                    metadata=meta,
                )
            )
        return samples

    def _from_hf(self, hf_id: str, cfg: dict) -> list[BenchmarkSample]:
        from datasets import load_dataset

        split = cfg.get("dataset_split", "test")
        print(f"GSM8K: loading HF {hf_id} split={split}")
        ds = load_dataset(hf_id, split=split)

        n = cfg.get("num_problems")
        if n is not None:
            ds = ds.select(range(min(n, len(ds))))

        if "question" not in ds.column_names or "answer" not in ds.column_names:
            raise ValueError(f"{hf_id} missing question/answer columns.")

        samples: list[BenchmarkSample] = []
        has_image = "image" in ds.column_names
        for i, row in enumerate(ds):
            img = None
            if has_image and row["image"] is not None:
                img = row["image"].convert("RGB")
            samples.append(
                BenchmarkSample(
                    question=str(row["question"]),
                    answer=str(row["answer"]),
                    image=img,
                    metadata={"id": row.get("id", i)},
                )
            )
        return samples


register_adapter(GSM8KAdapter())
