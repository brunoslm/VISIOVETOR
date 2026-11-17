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


def baixar_imagens_faltantes() -> None:
    """
    Baixa do dataset original as imagens do captions.jsonl que não estão em data/raw/images.

    Serve para publicar o app sem redistribuir as fotos do Flickr8k: vai só o captions.jsonl,
    e as imagens vêm do Hub na mesma ordem em que scripts/prepare_data.py as salvou.
    """
    if not config.ARQUIVO_LEGENDAS.exists():
        return
    registros = carregar_registros()
    if all(Path(r["path"]).exists() for r in registros):
        return

    from datasets import load_dataset

    config.DIR_IMAGENS.mkdir(parents=True, exist_ok=True)
    print(f"⬇️  Baixando as imagens do catálogo de {config.DATASET_HF}...")
    for divisao_hf, divisao in (("train", "train"), ("validation", "val")):
        da_divisao = [r for r in registros if r["split"] == divisao]
        if not da_divisao:
            continue
        ds = load_dataset(config.DATASET_HF, split=divisao_hf, streaming=True)
        for reg, ex in zip(da_divisao, ds.take(len(da_divisao))):
            legendas = ex["caption"] if isinstance(ex["caption"], list) else [ex["caption"]]
            legendas = [c.strip() for c in legendas if c and c.strip()]
            if legendas != reg["captions"]:
                raise RuntimeError(
                    f"O dataset {config.DATASET_HF} mudou: as legendas de {reg['image']} não batem com o captions.jsonl."
                )
            if not Path(reg["path"]).exists():
                img = ex["image"].convert("RGB")
                img.thumbnail((512, 512))  # mesmo tamanho máximo de scripts/prepare_data.py
                img.save(reg["path"], quality=90)
    print(f"✅ {len(registros)} imagens em {config.DIR_IMAGENS}")
