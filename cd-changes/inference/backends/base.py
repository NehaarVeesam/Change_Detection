"""Abstract inference backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class InferenceBackend(ABC):
    @property
    @abstractmethod
    def family(self) -> str:
        pass

    @abstractmethod
    def infer_pair(self, img_a: Path, img_b: Path) -> Dict[str, Any]:
        pass

    def warmup(self) -> None:
        pass
