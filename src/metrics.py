"""Métricas de recuperação (Recall@K) para avaliar o alinhamento imagem <-> texto."""

from __future__ import annotations

import numpy as np


def retrieval_recall(
    image_vecs: np.ndarray,
    text_vecs: np.ndarray,
    text_to_image: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """
    image_vecs: (N_img, D) normalizados; text_vecs: (N_txt, D) normalizados.
    text_to_image[j] = índice da imagem correta da legenda j.
    Retorna Recall@K para texto→imagem (t2i) e imagem→texto (i2t), em %.
    """
    sims = text_vecs @ image_vecs.T  # (N_txt, N_img)
    out: dict[str, float] = {}

    # texto -> imagem: posição da imagem correta no ranking de cada legenda
    correct = sims[np.arange(len(text_vecs)), text_to_image]
    t2i_rank = (sims > correct[:, None]).sum(axis=1)
    for k in ks:
        out[f"t2i_R@{k}"] = float((t2i_rank < k).mean() * 100)

    # imagem -> texto: melhor posição entre as legendas corretas da imagem
    sims_i = sims.T  # (N_img, N_txt)
    i2t_rank = np.empty(len(image_vecs))
    order = np.argsort(-sims_i, axis=1)
    for i in range(len(image_vecs)):
        positives = np.flatnonzero(text_to_image == i)
        i2t_rank[i] = np.flatnonzero(np.isin(order[i], positives))[0]
    for k in ks:
        out[f"i2t_R@{k}"] = float((i2t_rank < k).mean() * 100)
    return out


def format_metrics(m: dict[str, float]) -> str:
    return " | ".join(f"{k}: {v:5.1f}" for k, v in m.items())
