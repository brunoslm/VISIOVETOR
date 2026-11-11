"""Vetoriza o catálogo (imagens + legendas) e salva o cache em data/embeddings.

Uso:
  python scripts/build_index.py              # todos os modelos disponíveis
  python scripts/build_index.py --force      # reconstrói mesmo se houver cache
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.catalog import available_models, build_indexes  # noqa: E402
from src.embeddings import ClipEncoder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for label, path in available_models().items():
        print(f"\n=== {label}")
        images, captions = build_indexes(ClipEncoder(path), force=args.force)
        print(f"    {len(images)} imagens | {len(captions)} legendas | tags: {', '.join(images.all_tags())}")


if __name__ == "__main__":
    main()
