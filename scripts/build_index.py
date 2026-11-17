"""Vetoriza o catálogo (imagens + legendas) e salva o cache em data/embeddings.

Uso:
  python scripts/build_index.py              # todos os modelos disponíveis
  python scripts/build_index.py --force      # reconstrói mesmo se houver cache
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.catalog import modelos_disponiveis, construir_indices  # noqa: E402
from src.embeddings import CodificadorClip  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for rotulo, caminho in modelos_disponiveis().items():
        print(f"\n=== {rotulo}")
        imagens, legendas = construir_indices(CodificadorClip(caminho), forcar=args.force)
        print(f"    {len(imagens)} imagens | {len(legendas)} legendas | tags: {', '.join(imagens.todas_tags())}")


if __name__ == "__main__":
    main()
