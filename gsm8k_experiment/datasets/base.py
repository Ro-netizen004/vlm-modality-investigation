"""Dataset adapter protocol and registry."""

from __future__ import annotations

from typing import Protocol

from ..types import BenchmarkSample


class DatasetAdapter(Protocol):
    """Load a benchmark split into canonical samples."""

    dataset_type: str

    def load(self, cfg: dict) -> list[BenchmarkSample]:
        ...


_REGISTRY: dict[str, DatasetAdapter] = {}


def register_adapter(adapter: DatasetAdapter) -> None:
    _REGISTRY[adapter.dataset_type] = adapter


def get_adapter(dataset_type: str) -> DatasetAdapter:
    key = dataset_type.lower().strip()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown dataset_type={dataset_type!r}. Registered: {known}")
    return _REGISTRY[key]


def load_dataset(cfg: dict) -> list[BenchmarkSample]:
    """
    Load samples via the adapter selected by cfg['dataset_type'].

    cfg['dataset_name'] is accepted as a legacy alias for dataset_type when
    dataset_type is omitted and dataset_name has no '/' (not an HF repo id).
    """
    dtype = _resolve_dataset_type(cfg)
    cfg = {**cfg, "dataset_type": dtype}
    adapter = get_adapter(dtype)
    samples = adapter.load(cfg)
    print(f"Loaded {len(samples)} samples ({dtype})")
    if samples:
        print(f"  Sample Q: {samples[0].question[:100]} ...")
        print(f"  Sample A: {str(samples[0].answer)[-60:]}")
    return samples


def _resolve_dataset_type(cfg: dict) -> str:
    if cfg.get("dataset_type"):
        return str(cfg["dataset_type"]).lower().strip()

    name = cfg.get("dataset_name", "")
    if isinstance(name, str) and name and "/" not in name:
        return name.lower().strip()

    # HF hub id or rendered repo → default GSM8K adapter
    return "gsm8k"
