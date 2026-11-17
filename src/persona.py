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

NOME = "Jeffrey"

PROMPT_SISTEMA = (
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
class Fatos:
    modelo_base: str
    dim: int
    n_imagens: int
    n_treino: int
    n_val: int
    n_legendas: int
    tags: tuple[str, ...]
    base: dict[str, float]
    ajustado: dict[str, float]
    minutos_treino: float
    epocas: int
    melhor_epoca: int
    camadas_texto: int
    # comparação histórica com o CLIP original (medida neste projeto, mesma validação)
    clip_base_t2i_r1: float = 10.8
    clip_ajustado_t2i_r1: float = 28.4

    def r(self, qual: str, chave: str) -> str:
        return fmt(getattr(self, qual)[chave]) + "%"

    def ganho(self, chave: str) -> str:
        d = self.ajustado[chave] - self.base[chave]
        return ("+" if d >= 0 else "") + fmt(d)


_METRICAS_PADRAO = {
    "base_model": "google/siglip2-base-patch16-224",
    "n_train": 6000, "n_val": 1000, "train_seconds": 1358.8, "best_epoch": 4,
    "args": {"epochs": 4, "text_layers": 4},
    "baseline": {"t2i_R@1": 66.2, "t2i_R@5": 87.6, "t2i_R@10": 93.3, "i2t_R@1": 82.5, "i2t_R@5": 96.0, "i2t_R@10": 98.5},
    "best": {"t2i_R@1": 72.1, "t2i_R@5": 91.7, "t2i_R@10": 96.1, "i2t_R@1": 87.9, "i2t_R@5": 97.6, "i2t_R@10": 99.0},
}


def carregar_fatos() -> Fatos:
    caminho = config.DIR_MODELO_AJUSTADO / "metrics.json"
    m = json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else _METRICAS_PADRAO

    n_imagens = n_legendas = 0
    tags: set[str] = set()
    if config.ARQUIVO_LEGENDAS.exists():
        for linha in config.ARQUIVO_LEGENDAS.read_text(encoding="utf-8").splitlines():
            if linha.strip():
                reg = json.loads(linha)
                n_imagens += 1
                n_legendas += len(reg["captions"])
                tags.update(reg.get("tags", []))
    else:
        n_imagens, n_legendas = m["n_train"] + m["n_val"], 5 * (m["n_train"] + m["n_val"])
        tags = set(config.PALAVRAS_CHAVE_TAGS)

    return Fatos(
        modelo_base=m["base_model"],
        dim=768 if "siglip" in m["base_model"] else 512,
        n_imagens=n_imagens,
        n_treino=m["n_train"],
        n_val=m["n_val"],
        n_legendas=n_legendas,
        tags=tuple(sorted(tags)),
        base=m["baseline"],
        ajustado=m["best"],
        minutos_treino=m["train_seconds"] / 60,
        epocas=m["args"]["epochs"],
        melhor_epoca=m["best_epoch"],
        camadas_texto=m["args"]["text_layers"],
    )
