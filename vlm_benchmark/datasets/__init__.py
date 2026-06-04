"""Dataset adapters (import side-effects register adapters)."""

from . import gsm8k, svamp  # noqa: F401
from .base import get_adapter, load_dataset, register_adapter

__all__ = ["load_dataset", "get_adapter", "register_adapter", "gsm8k", "svamp"]
