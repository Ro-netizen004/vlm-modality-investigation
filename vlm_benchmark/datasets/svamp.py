"""SVAMP dataset adapter (text-only; images optional via metadata CSV later)."""

from __future__ import annotations

from ..types import BenchmarkSample
from .base import register_adapter


class SVAMPAdapter:
    dataset_type = "svamp"

    def load(self, cfg: dict) -> list[BenchmarkSample]:
        from datasets import load_dataset

        hf_id = cfg.get("hf_dataset_id", "ChilleD/SVAMP")
        split = cfg.get("dataset_split", "test")
        print(f"SVAMP: loading HF {hf_id}")
        ds = load_dataset(hf_id, split=split)

        n = cfg.get("num_problems")
        if n is not None:
            ds = ds.select(range(min(n, len(ds))))

        samples: list[BenchmarkSample] = []
        for i, row in enumerate(ds):
            # SVAMP schema: Body + Question → single prompt; Answer is numeric string
            body = row.get("Body", "")
            question_part = row.get("Question", row.get("question", ""))
            q = f"{body} {question_part}".strip() if body else str(question_part)
            answer = str(row.get("Answer", row.get("answer", "")))

            samples.append(
                BenchmarkSample(
                    question=q,
                    answer=answer,
                    image=None,
                    metadata={
                        "id": row.get("ID", i),
                        "equation": row.get("Equation"),
                        "type": row.get("Type"),
                    },
                )
            )
        return samples


register_adapter(SVAMPAdapter())
