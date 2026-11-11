"""Baixa um subconjunto do Flickr8k PT-BR do Hugging Face para data/raw.

Gera:
  data/raw/images/<arquivo>.jpg
  data/raw/captions.jsonl   (image, captions[5], tags, split)

Uso:
  python scripts/prepare_data.py                     # 2000 treino + 300 validação
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
from src.data import extract_tags  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=config.HF_DATASET)
    ap.add_argument("--n-train", type=int, default=config.N_TRAIN_IMAGES)
    ap.add_argument("--n-val", type=int, default=config.N_VAL_IMAGES)
    ap.add_argument("--max-side", type=int, default=512, help="redimensiona o maior lado (economiza disco)")
    args = ap.parse_args()

    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for hf_split, split, n in (("train", "train", args.n_train), ("validation", "val", args.n_val)):
        if n <= 0:
            continue
        # streaming=True evita baixar o dataset inteiro (~1.1 GB)
        ds = load_dataset(args.dataset, split=hf_split, streaming=True)
        for i, ex in enumerate(tqdm(ds.take(n), total=n, desc=f"Baixando {split}")):
            # o campo "filename" do dataset não é único -> nome pelo índice
            name = f"{split}_{i:05d}.jpg"
            img = ex["image"].convert("RGB")
            img.thumbnail((args.max_side, args.max_side))
            img.save(config.IMAGES_DIR / name, quality=90)
            captions = ex["caption"] if isinstance(ex["caption"], list) else [ex["caption"]]
            captions = [c.strip() for c in captions if c and c.strip()]
            records.append({
                "image": f"images/{name}",
                "captions": captions,
                "tags": extract_tags(captions),
                "split": split,
            })

    with open(config.CAPTIONS_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_tr = sum(r["split"] == "train" for r in records)
    print(f"✅ {len(records)} imagens ({n_tr} treino / {len(records) - n_tr} val) em {config.RAW_DIR}")


if __name__ == "__main__":
    main()
