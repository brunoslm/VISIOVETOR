"""Leitura do catálogo de imagens/legendas e extração de tags determinísticas."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import config


def _normalizar(texto: str) -> str:
    return unicodedata.normalize("NFC", texto.lower())


def extrair_tags(legendas: list[str]) -> list[str]:
    """Atribui tags por palavra-chave (match de palavra inteira) nas legendas."""
    palavras = set(re.findall(r"\w+", _normalizar(" ".join(legendas))))
    return sorted(tag for tag, pcs in config.PALAVRAS_CHAVE_TAGS.items() if palavras & {_normalizar(k) for k in pcs})


def carregar_registros(divisao: str | None = None, caminho: Path = config.ARQUIVO_LEGENDAS) -> list[dict]:
    """Cada registro: {"image": caminho relativo a data/raw, "captions": [...], "tags": [...], "split": ...}."""
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} não encontrado. Rode primeiro: python scripts/prepare_data.py"
        )
    registros = []
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            if linha.strip():
                reg = json.loads(linha)
                if divisao is None or reg.get("split") == divisao:
                    reg["path"] = str(config.DIR_BRUTOS / reg["image"])
                    registros.append(reg)
    return registros
