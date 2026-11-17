"""Métricas de recuperação (Recall@K) para avaliar o alinhamento imagem <-> texto."""

from __future__ import annotations

import numpy as np


def recall_recuperacao(
    vetores_imagem: np.ndarray,
    vetores_texto: np.ndarray,
    texto_para_imagem: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """
    vetores_imagem: (N_img, D) normalizados; vetores_texto: (N_txt, D) normalizados.
    texto_para_imagem[j] = índice da imagem correta da legenda j.
    Retorna Recall@K para texto→imagem (t2i) e imagem→texto (i2t), em %.
    """
    sims = vetores_texto @ vetores_imagem.T  # (N_txt, N_img)
    saida: dict[str, float] = {}

    # texto -> imagem: posição da imagem correta no ranking de cada legenda
    correta = sims[np.arange(len(vetores_texto)), texto_para_imagem]
    posicao_t2i = (sims > correta[:, None]).sum(axis=1)
    for k in ks:
        saida[f"t2i_R@{k}"] = float((posicao_t2i < k).mean() * 100)

    # imagem -> texto: melhor posição entre as legendas corretas da imagem
    sims_i = sims.T  # (N_img, N_txt)
    posicao_i2t = np.empty(len(vetores_imagem))
    ordem = np.argsort(-sims_i, axis=1)
    for i in range(len(vetores_imagem)):
        positivos = np.flatnonzero(texto_para_imagem == i)
        posicao_i2t[i] = np.flatnonzero(np.isin(ordem[i], positivos))[0]
    for k in ks:
        saida[f"i2t_R@{k}"] = float((posicao_i2t < k).mean() * 100)
    return saida


def formatar_metricas(m: dict[str, float]) -> str:
    return " | ".join(f"{k}: {v:5.1f}" for k, v in m.items())
