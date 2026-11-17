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
from src.data import carregar_registros  # noqa: E402
from src.embeddings import CodificadorClip, normalizar_l2  # noqa: E402
from src.metrics import formatar_metricas, recall_recuperacao  # noqa: E402


def fixar_semente(semente: int) -> None:
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)


def codificar_legendas(cod: CodificadorClip, legendas: list[str]) -> torch.Tensor:
    return normalizar_l2(cod.features_texto(**cod.tokenizar(legendas)))


@torch.no_grad()
def avaliar(cod: CodificadorClip, vetores_img: torch.Tensor, registros: list[dict]) -> dict[str, float]:
    cod.modelo.eval()
    legs, dona = [], []
    for i, r in enumerate(registros):
        legs.extend(r["captions"])
        dona.extend([i] * len(r["captions"]))
    txt = torch.cat([codificar_legendas(cod, legs[j : j + 128]).cpu() for j in range(0, len(legs), 128)]).numpy()
    return recall_recuperacao(vetores_img.numpy(), txt, np.array(dona))


def rsum(m: dict[str, float]) -> float:
    return sum(v for k, v in m.items() if "R@" in k)


def configurar_treinaveis(cod: CodificadorClip, camadas_texto: int) -> tuple[list, list]:
    """Congela tudo e libera as últimas camadas do texto + cabeça + temperatura."""
    m = cod.modelo
    for p in m.parameters():
        p.requires_grad = False

    texto_modelo = m.text_model
    camadas = texto_modelo.encoder.layers
    n = min(camadas_texto, len(camadas))
    params_codificador = [p for camada in camadas[len(camadas) - n :] for p in camada.parameters()]
    # tudo do texto que fica depois do encoder (layer norm final, head de pooling)
    for nome, modulo in texto_modelo.named_children():
        if nome not in ("embeddings", "encoder"):
            params_codificador += list(modulo.parameters())
    if hasattr(m, "text_projection"):  # CLIP
        params_codificador += list(m.text_projection.parameters())
    escalares = [m.logit_scale] + ([m.logit_bias] if getattr(m, "logit_bias", None) is not None else [])

    for p in params_codificador + escalares:
        p.requires_grad = True
    n_treino = sum(p.numel() for p in params_codificador + escalares)
    n_total = sum(p.numel() for p in m.parameters())
    print(f"🧮 Parâmetros treináveis: {n_treino / 1e6:.1f}M de {n_total / 1e6:.1f}M ({100 * n_treino / n_total:.1f}%)")
    return params_codificador, escalares


def perda_contrastiva(cod: CodificadorClip, txt: torch.Tensor, img: torch.Tensor) -> tuple[torch.Tensor, float]:
    m = cod.modelo
    escala = m.logit_scale.exp()
    logits = escala * txt @ img.T
    if cod.e_siglip:
        logits = logits + m.logit_bias
        rotulos = 2 * torch.eye(len(txt), device=txt.device) - 1
        perda = -F.logsigmoid(rotulos * logits).sum() / len(txt)
    else:
        alvo = torch.arange(len(txt), device=txt.device)
        perda = 0.5 * (F.cross_entropy(logits, alvo) + F.cross_entropy(logits.T, alvo))
    return perda, escala.item()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=config.MODELO_BASE)
    ap.add_argument("--output", default=str(config.DIR_MODELO_AJUSTADO))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5, help="LR das camadas de texto")
    ap.add_argument("--scalar-lr", type=float, default=1e-4, help="LR da temperatura/viés")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--text-layers", type=int, default=4, help="quantas camadas finais do texto treinar")
    ap.add_argument("--warmup", type=float, default=0.1, help="fração de passos de warmup")
    ap.add_argument("--seed", type=int, default=config.SEMENTE)
    args = ap.parse_args()

    fixar_semente(args.seed)
    treino = carregar_registros("train")
    val = carregar_registros("val")
    if not treino or not val:
        sys.exit("❌ É preciso ter registros com split 'train' e 'val' em data/raw/captions.jsonl")
    print(f"📚 {len(treino)} imagens de treino / {len(val)} de validação | modelo={args.base_model} | device={config.DISPOSITIVO}")

    cod = CodificadorClip(args.base_model)
    # visão congelada: vetores de imagem calculados uma vez só
    img_treino = torch.from_numpy(cod.codificar_imagens([r["path"] for r in treino], mostrar_progresso=True))
    img_val = torch.from_numpy(cod.codificar_imagens([r["path"] for r in val], mostrar_progresso=True))

    linha_base = avaliar(cod, img_val, val)
    print(f"\n📏 Baseline (antes do fine-tuning): {formatar_metricas(linha_base)}  | rsum={rsum(linha_base):.1f}\n")

    params_texto, escalares = configurar_treinaveis(cod, args.text_layers)
    otimizador = torch.optim.AdamW(
        [
            {"params": params_texto, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": escalares, "lr": args.scalar_lr, "weight_decay": 0.0},
        ]
    )
    passos_por_epoca = math.ceil(len(treino) / args.batch_size)
    total_passos = passos_por_epoca * args.epochs
    aquecimento = max(1, int(args.warmup * total_passos))

    def fator_lr(passo: int) -> float:
        if passo < aquecimento:
            return (passo + 1) / aquecimento
        return 0.5 * (1 + math.cos(math.pi * (passo - aquecimento) / max(1, total_passos - aquecimento)))

    agendador = torch.optim.lr_scheduler.LambdaLR(otimizador, fator_lr)

    melhor = rsum(linha_base)
    historico = [{"epoch": 0, **linha_base}]
    dir_saida = Path(args.output)
    salvo = False
    t0 = time.time()

    for epoca in range(1, args.epochs + 1):
        cod.modelo.train()
        ordem = list(range(len(treino)))
        random.shuffle(ordem)
        barra = tqdm(range(0, len(ordem), args.batch_size), desc=f"Época {epoca}/{args.epochs}", unit="passo")
        acumulado, passos = 0.0, 0
        for i in barra:
            idx = ordem[i : i + args.batch_size]
            if len(idx) < 2:
                continue
            # uma legenda aleatória (das 5) por imagem a cada época = data augmentation textual
            legs = [random.choice(treino[j]["captions"]) for j in idx]
            img = img_treino[idx].to(cod.dispositivo)
            txt = codificar_legendas(cod, legs)
            perda, escala = perda_contrastiva(cod, txt, img)

            otimizador.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(params_texto, 1.0)
            otimizador.step()
            agendador.step()

            acumulado += perda.item()
            passos += 1
            barra.set_postfix(loss=f"{acumulado / passos:.3f}", temp=f"{escala:.1f}")

        metricas = avaliar(cod, img_val, val)
        historico.append({"epoch": epoca, "train_loss": acumulado / max(1, passos), **metricas})
        marca = ""
        if rsum(metricas) > melhor:
            melhor = rsum(metricas)
            cod.modelo.save_pretrained(dir_saida)
            cod.processador.save_pretrained(dir_saida)
            salvo = True
            marca = "  💾 melhor até agora — salvo"
        print(f"   val: {formatar_metricas(metricas)} | rsum={rsum(metricas):.1f}{marca}")

    decorrido = time.time() - t0
    if not salvo:
        print("\n⚠️  Nenhuma época superou o modelo base; nada foi salvo. Tente outro --lr / --epochs.")
        return

    melhor_ep = max(historico[1:], key=rsum)
    relatorio = {
        "base_model": args.base_model,
        "args": vars(args),
        "n_train": len(treino),
        "n_val": len(val),
        "train_seconds": round(decorrido, 1),
        "baseline": linha_base,
        "best": {k: v for k, v in melhor_ep.items() if "R@" in k},
        "best_epoch": melhor_ep["epoch"],
        "history": historico,
    }
    (dir_saida / "metrics.json").write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✅ Fine-tuning concluído em {decorrido / 60:.1f} min. Modelo salvo em {dir_saida}")
    print(f"{'métrica':<10} {'base':>7} {'ajustado':>9} {'Δ':>7}")
    for k, v in linha_base.items():
        print(f"{k:<10} {v:7.1f} {relatorio['best'][k]:9.1f} {relatorio['best'][k] - v:+7.1f}")


if __name__ == "__main__":
    main()
