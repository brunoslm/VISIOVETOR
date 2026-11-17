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
class ResultadoBusca:
    indice: int
    score: float
    metadados: dict


@dataclass
class IndiceVetorial:
    vetores: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    metadados: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.vetores) != len(self.metadados):
            raise ValueError("vetores e metadados devem ter o mesmo tamanho")
        self._mascaras_tags: dict[str, np.ndarray] | None = None

    def _mascara_tags(self, tags: list[str] | None) -> np.ndarray:
        """Máscara booleana (N,) dos itens que têm TODAS as tags. Máscaras por tag ficam em cache."""
        mascara = np.ones(len(self), dtype=bool)
        if not tags:
            return mascara
        if self._mascaras_tags is None:
            mascaras: dict[str, np.ndarray] = {}
            for i, m in enumerate(self.metadados):
                for t in m.get("tags", []):
                    mascaras.setdefault(t, np.zeros(len(self), dtype=bool))[i] = True
            self._mascaras_tags = mascaras
        for t in set(tags):
            mascara &= self._mascaras_tags.get(t, np.zeros(len(self), dtype=bool))
        return mascara

    def __len__(self) -> int:
        return len(self.metadados)

    # -- escrita -------------------------------------------------------------
    def adicionar(self, vetores: np.ndarray, metadados: list[dict]) -> None:
        if len(vetores) != len(metadados):
            raise ValueError("vetores e metadados devem ter o mesmo tamanho")
        vetores = vetores.astype(np.float32)
        vetores /= np.linalg.norm(vetores, axis=1, keepdims=True).clip(min=1e-12)
        self.vetores = np.vstack([self.vetores, vetores]) if len(self) else vetores
        self.metadados.extend(metadados)
        self._mascaras_tags = None  # catálogo mudou: refaz as máscaras na próxima busca

    # -- leitura -------------------------------------------------------------
    def buscar(
        self,
        consulta: np.ndarray,
        top_k: int = 8,
        tags: list[str] | None = None,
        excluir: set[int] | None = None,
    ) -> list[ResultadoBusca]:
        """Busca Top-K por cosseno, com filtro determinístico opcional (AND entre tags)."""
        if not len(self):
            return []
        q = consulta.astype(np.float32).reshape(-1)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        scores = self.vetores @ q  # (N,)

        mascara = self._mascara_tags(tags)
        if excluir:
            mascara[list(excluir)] = False

        candidatos = np.flatnonzero(mascara)
        if candidatos.size == 0:
            return []
        k = min(top_k, candidatos.size)
        topo = candidatos[np.argpartition(-scores[candidatos], k - 1)[:k]]
        topo = topo[np.argsort(-scores[topo])]
        return [ResultadoBusca(int(i), float(scores[i]), self.metadados[i]) for i in topo]

    def contar_com_tags(self, tags: list[str] | None) -> int:
        """Quantos itens passam pelo filtro de tags (todas precisam estar presentes)."""
        return int(self._mascara_tags(tags).sum())

    def todas_tags(self) -> list[str]:
        return sorted({t for m in self.metadados for t in m.get("tags", [])})

    # -- persistência --------------------------------------------------------
    def salvar(self, prefixo: str | Path) -> None:
        p = Path(prefixo)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p.with_suffix(".npy"), self.vetores)
        p.with_suffix(".json").write_text(json.dumps(self.metadados, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def carregar(classe, prefixo: str | Path) -> "VectorStore":
        p = Path(prefixo)
        vetores = np.load(p.with_suffix(".npy"))
        metadados = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        return classe(vetores=vetores, metadados=metadados)

    @staticmethod
    def existe(prefixo: str | Path) -> bool:
        p = Path(prefixo)
        return p.with_suffix(".npy").exists() and p.with_suffix(".json").exists()
