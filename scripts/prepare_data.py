"""Baixa um subconjunto do Flickr8k PT-BR do Hugging Face para data/raw.

Gera:
  data/raw/images/<arquivo>.jpg
  data/raw/captions.jsonl   (image, captions[5], tags, split)

Uso:
  python scripts/prepare_data.py                     # 6000 treino + 1000 validação (Flickr8k completo)
  python scripts/prepare_data.py --n-train 500 --n-val 100

Também é possível usar suas próprias imagens: basta montar o captions.jsonl
no mesmo formato (split "train" para fine-tuning, "val" para avaliação).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

import config  # noqa: E402
from src.data import extrair_tags  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=config.DATASET_HF)
    ap.add_argument("--n-train", type=int, default=config.N_IMAGENS_TREINO)
    ap.add_argument("--n-val", type=int, default=config.N_IMAGENS_VAL)
    ap.add_argument("--max-side", type=int, default=512, help="redimensiona o maior lado (economiza disco)")
    args = ap.parse_args()

    config.DIR_IMAGENS.mkdir(parents=True, exist_ok=True)
    registros = []
    for divisao_hf, divisao, n in (("train", "train", args.n_train), ("validation", "val", args.n_val)):
        if n <= 0:
            continue
        # streaming=True evita baixar o dataset inteiro (~1.1 GB)
        ds = load_dataset(args.dataset, split=divisao_hf, streaming=True)
        for i, ex in enumerate(tqdm(ds.take(n), total=n, desc=f"Baixando {divisao}")):
            # o campo "filename" do dataset não é único -> nome pelo índice
            nome_arquivo = f"{divisao}_{i:05d}.jpg"
            img = ex["image"].convert("RGB")
            img.thumbnail((args.max_side, args.max_side))
            img.save(config.DIR_IMAGENS / nome_arquivo, quality=90)
            legendas = ex["caption"] if isinstance(ex["caption"], list) else [ex["caption"]]
            legendas = [c.strip() for c in legendas if c and c.strip()]
            registros.append({
                "image": f"images/{nome_arquivo}",
                "captions": legendas,
                "tags": extrair_tags(legendas),
                "split": divisao,
            })

    with open(config.ARQUIVO_LEGENDAS, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_tr = sum(r["split"] == "train" for r in registros)
    print(f"✅ {len(registros)} imagens ({n_tr} treino / {len(registros) - n_tr} val) em {config.DIR_BRUTOS}")


if __name__ == "__main__":
    main()
