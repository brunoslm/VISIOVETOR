"""Engine de busca vetorial em memória (NumPy) com filtros híbridos por tag.

Ranking: S = Q · Vᵀ  (vetores já normalizados ⇒ similaridade de cosseno).
Complexidade: O(N · D).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SearchResult:
    index: int
    score: float
    metadata: dict


@dataclass
class VectorStore:
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    metadata: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.vectors) != len(self.metadata):
            raise ValueError("vectors e metadata devem ter o mesmo tamanho")

    def __len__(self) -> int:
        return len(self.metadata)

    # -- escrita -------------------------------------------------------------
    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        if len(vectors) != len(metadata):
            raise ValueError("vectors e metadata devem ter o mesmo tamanho")
        vectors = vectors.astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
        self.vectors = np.vstack([self.vectors, vectors]) if len(self) else vectors
        self.metadata.extend(metadata)

    # -- leitura -------------------------------------------------------------
    def search(
        self,
        query: np.ndarray,
        top_k: int = 8,
        tags: list[str] | None = None,
        exclude: set[int] | None = None,
    ) -> list[SearchResult]:
        """Busca Top-K por cosseno, com filtro determinístico opcional (AND entre tags)."""
        if not len(self):
            return []
        q = query.astype(np.float32).reshape(-1)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        scores = self.vectors @ q  # (N,)

        mask = np.ones(len(self), dtype=bool)
        if tags:
            wanted = set(tags)
            mask &= np.array([wanted.issubset(m.get("tags", [])) for m in self.metadata])
        if exclude:
            mask[list(exclude)] = False

        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return []
        k = min(top_k, candidates.size)
        top = candidates[np.argpartition(-scores[candidates], k - 1)[:k]]
        top = top[np.argsort(-scores[top])]
        return [SearchResult(int(i), float(scores[i]), self.metadata[i]) for i in top]

    def count_with_tags(self, tags: list[str] | None) -> int:
        """Quantos itens passam pelo filtro de tags (todas precisam estar presentes)."""
        if not tags:
            return len(self)
        wanted = set(tags)
        return sum(wanted.issubset(m.get("tags", [])) for m in self.metadata)

    def all_tags(self) -> list[str]:
        return sorted({t for m in self.metadata for t in m.get("tags", [])})

    # -- persistência --------------------------------------------------------
    def save(self, path_prefix: str | Path) -> None:
        p = Path(path_prefix)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p.with_suffix(".npy"), self.vectors)
        p.with_suffix(".json").write_text(json.dumps(self.metadata, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path_prefix: str | Path) -> "VectorStore":
        p = Path(path_prefix)
        vectors = np.load(p.with_suffix(".npy"))
        metadata = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(vectors=vectors, metadata=metadata)

    @staticmethod
    def exists(path_prefix: str | Path) -> bool:
        p = Path(path_prefix)
        return p.with_suffix(".npy").exists() and p.with_suffix(".json").exists()
