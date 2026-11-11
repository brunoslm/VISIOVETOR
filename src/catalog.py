"""Construção/carregamento dos índices vetoriais (imagens e legendas) por modelo."""

from __future__ import annotations

import re
from pathlib import Path

import config
from src.data import load_records
from src.embeddings import ClipEncoder
from src.vector_store import VectorStore


def available_models() -> dict[str, str]:
    """Rótulo amigável -> caminho/ID do modelo."""
    models = {f"Base · {config.BASE_MODEL}": config.BASE_MODEL}
    if (config.FINETUNED_MODEL_DIR / "config.json").exists():
        models[f"Ajustado em PT-BR · {config.FINETUNED_MODEL_DIR.relative_to(config.ROOT_DIR)}"] = str(config.FINETUNED_MODEL_DIR)
    return models


def _slug(model_path: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", Path(model_path).name if Path(model_path).exists() else model_path).strip("_")


def index_prefix(model_path: str, kind: str) -> Path:
    return config.EMBEDDINGS_DIR / f"{_slug(model_path)}__{kind}"


def build_indexes(encoder: ClipEncoder, force: bool = False) -> tuple[VectorStore, VectorStore]:
    """Retorna (índice de imagens, índice de legendas), usando cache em data/embeddings."""
    img_prefix = index_prefix(encoder.model_path, "images")
    cap_prefix = index_prefix(encoder.model_path, "captions")
    records = load_records()

    if not force and VectorStore.exists(img_prefix) and VectorStore.exists(cap_prefix):
        images, captions = VectorStore.load(img_prefix), VectorStore.load(cap_prefix)
        # modelo local re-treinado depois do índice => índice velho
        weights = list(Path(encoder.model_path).glob("*.safetensors")) if Path(encoder.model_path).is_dir() else []
        index_time = img_prefix.with_suffix(".npy").stat().st_mtime
        stale_model = any(w.stat().st_mtime > index_time for w in weights)
        if len(images) == len(records) and not stale_model:
            return images, captions
        print("⚠️  Catálogo ou modelo mudou desde o último índice — reconstruindo.")

    print(f"🔧 Construindo índices para {encoder.model_path} ({len(records)} imagens)...")
    img_meta = [
        {"image": r["image"], "path": r["path"], "caption": r["captions"][0], "tags": r["tags"], "split": r["split"]}
        for r in records
    ]
    images = VectorStore()
    images.add(encoder.encode_images([r["path"] for r in records], show_progress=True), img_meta)

    cap_meta = [
        {"caption": c, "image": r["image"], "tags": r["tags"]}
        for r in records
        for c in r["captions"]
    ]
    captions = VectorStore()
    captions.add(encoder.encode_texts([m["caption"] for m in cap_meta], show_progress=True), cap_meta)

    images.save(img_prefix)
    captions.save(cap_prefix)
    print(f"✅ Índices salvos em {config.EMBEDDINGS_DIR}")
    return images, captions
