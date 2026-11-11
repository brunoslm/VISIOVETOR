"""Leitura do catálogo de imagens/legendas e extração de tags determinísticas."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import config


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.lower())


def extract_tags(captions: list[str]) -> list[str]:
    """Atribui tags por palavra-chave (match de palavra inteira) nas legendas."""
    words = set(re.findall(r"\w+", _normalize(" ".join(captions))))
    return sorted(tag for tag, kws in config.TAG_KEYWORDS.items() if words & {_normalize(k) for k in kws})


def load_records(split: str | None = None, path: Path = config.CAPTIONS_FILE) -> list[dict]:
    """Cada registro: {"image": caminho relativo a data/raw, "captions": [...], "tags": [...], "split": ...}."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Rode primeiro: python scripts/prepare_data.py"
        )
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if split is None or rec.get("split") == split:
                    rec["path"] = str(config.RAW_DIR / rec["image"])
                    records.append(rec)
    return records
