"""Shared types for the benchmark framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass
class BenchmarkSample:
    """Canonical record produced by every dataset adapter."""

    question: str
    answer: str
    image: Image.Image | None
    metadata: dict[str, Any] = field(default_factory=dict)
