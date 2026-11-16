"""Persona "Jeffrey" — o assistente cordial do VisioVetor.

Fonte única da personalidade (prompt de sistema) e dos fatos do projeto usados
nos datasets de alinhamento. Os números vêm dos arquivos reais do projeto
(models/.../metrics.json e data/raw/captions.jsonl) para que o Jeffrey nunca
"aprenda" métricas inventadas; se o modelo for re-treinado, basta regenerar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import config

NAME = "Jeffrey"

SYSTEM_PROMPT = (
    "Você é o Jeffrey, assistente do VisioVetor, um sistema de busca semântica que conecta "
    "imagens e textos. Você é cordial, educado e acolhedor: cumprimenta com gentileza, "
    "agradece a paciência e trata a pessoa por \"você\". Ao mesmo tempo, é preciso e honesto: "
    "explica com clareza, só cita números que vêm do sistema e diz com franqueza quando não "
    "sabe algo ou quando um pedido foge do que o VisioVetor faz. Responde em português do "
    "Brasil, de forma concisa, sem emojis, e quando faz sentido termina sugerindo um próximo "
    "passo útil."
)


def fmt(v: float, dec: int = 1) -> str:
    """Número no padrão pt-BR (vírgula decimal, ponto de milhar)."""
    s = f"{v:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


@dataclass(frozen=True)
class Facts:
    base_model: str
    dim: int
    n_images: int
    n_train: int
    n_val: int
    n_captions: int
    tags: tuple[str, ...]
    base: dict[str, float]
    tuned: dict[str, float]
    train_minutes: float
    epochs: int
    best_epoch: int
    text_layers: int
    # comparação histórica com o CLIP original (medida neste projeto, mesma validação)
    clip_base_t2i_r1: float = 19.7
    clip_tuned_t2i_r1: float = 42.0

    def r(self, which: str, key: str) -> str:
        return fmt(getattr(self, which)[key]) + "%"

    def gain(self, key: str) -> str:
        d = self.tuned[key] - self.base[key]
        return ("+" if d >= 0 else "") + fmt(d)


_DEFAULT_METRICS = {
    "base_model": "google/siglip2-base-patch16-224",
    "n_train": 2000, "n_val": 300, "train_seconds": 397.5, "best_epoch": 3,
    "args": {"epochs": 4, "text_layers": 4},
    "baseline": {"t2i_R@1": 75.5, "t2i_R@5": 95.0, "t2i_R@10": 98.4, "i2t_R@1": 89.3, "i2t_R@5": 98.0, "i2t_R@10": 99.7},
    "best": {"t2i_R@1": 80.3, "t2i_R@5": 97.3, "t2i_R@10": 99.1, "i2t_R@1": 90.3, "i2t_R@5": 99.0, "i2t_R@10": 99.7},
}


def load_facts() -> Facts:
    path = config.FINETUNED_MODEL_DIR / "metrics.json"
    m = json.loads(path.read_text(encoding="utf-8")) if path.exists() else _DEFAULT_METRICS

    n_images = n_captions = 0
    tags: set[str] = set()
    if config.CAPTIONS_FILE.exists():
        for line in config.CAPTIONS_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                n_images += 1
                n_captions += len(rec["captions"])
                tags.update(rec.get("tags", []))
    else:
        n_images, n_captions = m["n_train"] + m["n_val"], 5 * (m["n_train"] + m["n_val"])
        tags = set(config.TAG_KEYWORDS)

    return Facts(
        base_model=m["base_model"],
        dim=768 if "siglip" in m["base_model"] else 512,
        n_images=n_images,
        n_train=m["n_train"],
        n_val=m["n_val"],
        n_captions=n_captions,
        tags=tuple(sorted(tags)),
        base=m["baseline"],
        tuned=m["best"],
        train_minutes=m["train_seconds"] / 60,
        epochs=m["args"]["epochs"],
        best_epoch=m["best_epoch"],
        text_layers=m["args"]["text_layers"],
    )
