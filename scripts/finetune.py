"""Fine-tuning contrastivo de um modelo imagem-texto (SigLIP 2 ou CLIP) em legendas PT-BR.

Estratégia "leve", pensada para rodar em CPU em poucos minutos:
  * O encoder de visão fica CONGELADO; os vetores de imagem são calculados uma
    única vez e mantidos em cache.
  * Treinamos apenas: as últimas N camadas do encoder de texto, a normalização final
    e a cabeça/projeção do texto, e a temperatura (e o viés, no SigLIP).
  * Perda:
      - SigLIP (sigmoide):  L = -1/B Σᵢⱼ log σ(zᵢⱼ · (τ·tᵢ·iⱼ + b)),  zᵢⱼ = +1 se i=j, senão -1
      - CLIP  (InfoNCE):    L = ½ [CE(S, y) + CE(Sᵀ, y)],  S = τ · T̂ · Îᵀ

Assim o modelo aprende o vocabulário e o estilo das legendas em português sem
perder o espaço visual pré-treinado. O melhor checkpoint na validação é salvo
só se superar o modelo base.

Uso:
  python scripts/finetune.py
  python scripts/finetune.py --epochs 4 --text-layers 4 --lr 1e-5
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tqdm import tqdm  # noqa: E402

import config  # noqa: E402
from src.data import load_records  # noqa: E402
from src.embeddings import ClipEncoder, l2_normalize  # noqa: E402
from src.metrics import format_metrics, retrieval_recall  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_captions(enc: ClipEncoder, captions: list[str]) -> torch.Tensor:
    return l2_normalize(enc.text_features(**enc.tokenize(captions)))


@torch.no_grad()
def evaluate(enc: ClipEncoder, img_vecs: torch.Tensor, records: list[dict]) -> dict[str, float]:
    enc.model.eval()
    caps, owner = [], []
    for i, r in enumerate(records):
        caps.extend(r["captions"])
        owner.extend([i] * len(r["captions"]))
    txt = torch.cat([encode_captions(enc, caps[j : j + 128]).cpu() for j in range(0, len(caps), 128)]).numpy()
    return retrieval_recall(img_vecs.numpy(), txt, np.array(owner))


def rsum(m: dict[str, float]) -> float:
    return sum(v for k, v in m.items() if "R@" in k)


def configure_trainable(enc: ClipEncoder, text_layers: int) -> tuple[list, list]:
    """Congela tudo e libera as últimas camadas do texto + cabeça + temperatura."""
    m = enc.model
    for p in m.parameters():
        p.requires_grad = False

    text = m.text_model
    layers = text.encoder.layers
    n = min(text_layers, len(layers))
    encoder_params = [p for layer in layers[len(layers) - n :] for p in layer.parameters()]
    # tudo do texto que fica depois do encoder (layer norm final, head de pooling)
    for name, module in text.named_children():
        if name not in ("embeddings", "encoder"):
            encoder_params += list(module.parameters())
    if hasattr(m, "text_projection"):  # CLIP
        encoder_params += list(m.text_projection.parameters())
    scalars = [m.logit_scale] + ([m.logit_bias] if getattr(m, "logit_bias", None) is not None else [])

    for p in encoder_params + scalars:
        p.requires_grad = True
    n_train = sum(p.numel() for p in encoder_params + scalars)
    n_total = sum(p.numel() for p in m.parameters())
    print(f"🧮 Parâmetros treináveis: {n_train / 1e6:.1f}M de {n_total / 1e6:.1f}M ({100 * n_train / n_total:.1f}%)")
    return encoder_params, scalars


def contrastive_loss(enc: ClipEncoder, txt: torch.Tensor, img: torch.Tensor) -> tuple[torch.Tensor, float]:
    m = enc.model
    scale = m.logit_scale.exp()
    logits = scale * txt @ img.T
    if enc.is_siglip:
        logits = logits + m.logit_bias
        labels = 2 * torch.eye(len(txt), device=txt.device) - 1
        loss = -F.logsigmoid(labels * logits).sum() / len(txt)
    else:
        target = torch.arange(len(txt), device=txt.device)
        loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
    return loss, scale.item()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=config.BASE_MODEL)
    ap.add_argument("--output", default=str(config.FINETUNED_MODEL_DIR))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5, help="LR das camadas de texto")
    ap.add_argument("--scalar-lr", type=float, default=1e-4, help="LR da temperatura/viés")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--text-layers", type=int, default=4, help="quantas camadas finais do texto treinar")
    ap.add_argument("--warmup", type=float, default=0.1, help="fração de passos de warmup")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    set_seed(args.seed)
    train = load_records("train")
    val = load_records("val")
    if not train or not val:
        sys.exit("❌ É preciso ter registros com split 'train' e 'val' em data/raw/captions.jsonl")
    print(f"📚 {len(train)} imagens de treino / {len(val)} de validação | modelo={args.base_model} | device={config.DEVICE}")

    enc = ClipEncoder(args.base_model)
    # visão congelada: vetores de imagem calculados uma vez só
    train_img = torch.from_numpy(enc.encode_images([r["path"] for r in train], show_progress=True))
    val_img = torch.from_numpy(enc.encode_images([r["path"] for r in val], show_progress=True))

    baseline = evaluate(enc, val_img, val)
    print(f"\n📏 Baseline (antes do fine-tuning): {format_metrics(baseline)}  | rsum={rsum(baseline):.1f}\n")

    text_params, scalars = configure_trainable(enc, args.text_layers)
    opt = torch.optim.AdamW(
        [
            {"params": text_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": scalars, "lr": args.scalar_lr, "weight_decay": 0.0},
        ]
    )
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(args.warmup * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best = rsum(baseline)
    history = [{"epoch": 0, **baseline}]
    out_dir = Path(args.output)
    saved = False
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        enc.model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        pbar = tqdm(range(0, len(order), args.batch_size), desc=f"Época {epoch}/{args.epochs}", unit="passo")
        running, steps = 0.0, 0
        for i in pbar:
            idx = order[i : i + args.batch_size]
            if len(idx) < 2:
                continue
            # uma legenda aleatória (das 5) por imagem a cada época = data augmentation textual
            caps = [random.choice(train[j]["captions"]) for j in idx]
            img = train_img[idx].to(enc.device)
            txt = encode_captions(enc, caps)
            loss, scale = contrastive_loss(enc, txt, img)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(text_params, 1.0)
            opt.step()
            sched.step()

            running += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{running / steps:.3f}", temp=f"{scale:.1f}")

        metrics = evaluate(enc, val_img, val)
        history.append({"epoch": epoch, "train_loss": running / max(1, steps), **metrics})
        flag = ""
        if rsum(metrics) > best:
            best = rsum(metrics)
            enc.model.save_pretrained(out_dir)
            enc.processor.save_pretrained(out_dir)
            saved = True
            flag = "  💾 melhor até agora — salvo"
        print(f"   val: {format_metrics(metrics)} | rsum={rsum(metrics):.1f}{flag}")

    elapsed = time.time() - t0
    if not saved:
        print("\n⚠️  Nenhuma época superou o modelo base; nada foi salvo. Tente outro --lr / --epochs.")
        return

    best_ep = max(history[1:], key=rsum)
    report = {
        "base_model": args.base_model,
        "args": vars(args),
        "n_train": len(train),
        "n_val": len(val),
        "train_seconds": round(elapsed, 1),
        "baseline": baseline,
        "best": {k: v for k, v in best_ep.items() if "R@" in k},
        "best_epoch": best_ep["epoch"],
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✅ Fine-tuning concluído em {elapsed / 60:.1f} min. Modelo salvo em {out_dir}")
    print(f"{'métrica':<10} {'base':>7} {'ajustado':>9} {'Δ':>7}")
    for k, v in baseline.items():
        print(f"{k:<10} {v:7.1f} {report['best'][k]:9.1f} {report['best'][k] - v:+7.1f}")


if __name__ == "__main__":
    main()
