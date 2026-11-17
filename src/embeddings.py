"""Pipeline de inferência imagem <-> texto (PyTorch + Hugging Face) com normalização L2.

Funciona com modelos da família CLIP (OpenAI CLIP, SigLIP, SigLIP 2): imagem e
texto são projetados no mesmo espaço latente. Depois da normalização L2,
similaridade de cosseno == produto escalar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

import config


def normalizar_l2(x: torch.Tensor) -> torch.Tensor:
    """v̂ = v / ||v||₂ (por linha)."""
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)


def carregar_imagem(caminho: str | Path) -> Image.Image:
    return Image.open(caminho).convert("RGB")


def _em_lotes(itens: Sequence, tamanho: int) -> Iterable[Sequence]:
    for i in range(0, len(itens), tamanho):
        yield itens[i : i + tamanho]


def _como_tensor(saida) -> torch.Tensor:
    """get_*_features devolve tensor (transformers 4.x) ou um objeto com pooler_output (5.x)."""
    return saida if torch.is_tensor(saida) else saida.pooler_output


class CodificadorClip:
    """Encoder de imagens e textos para modelos da família CLIP/SigLIP."""

    def __init__(self, caminho_modelo: str | Path = config.MODELO_BASE, dispositivo: str = config.DISPOSITIVO):
        self.caminho_modelo = str(caminho_modelo)
        self.dispositivo = dispositivo
        self.modelo = AutoModel.from_pretrained(self.caminho_modelo).to(dispositivo).eval()
        self.processador = AutoProcessor.from_pretrained(self.caminho_modelo)
        self.tokenizador = getattr(self.processador, "tokenizer", self.processador)
        # SigLIP foi treinado com texto em minúsculas, padding fixo de 64 e perda sigmoide
        self.e_siglip = "siglip" in self.modelo.config.model_type

    @property
    def dim_embedding(self) -> int:
        return int(self.codificar_textos(["teste"]).shape[1])

    # -- tokenização / features brutas (usadas também no fine-tuning) -------
    def tokenizar(self, textos: Sequence[str]) -> dict[str, torch.Tensor]:
        if self.e_siglip:
            textos = [t.lower() for t in textos]
            tok = self.tokenizador(list(textos), padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        else:
            tok = self.tokenizador(list(textos), padding=True, max_length=77, truncation=True, return_tensors="pt")
        return {k: v.to(self.dispositivo) for k, v in tok.items() if k in ("input_ids", "attention_mask")}

    def valores_pixel(self, imagens: Sequence[Image.Image]) -> dict[str, torch.Tensor]:
        entradas = self.processador(images=list(imagens), return_tensors="pt")
        return {k: v.to(self.dispositivo) for k, v in entradas.items() if k != "input_ids"}

    def features_imagem(self, **entradas_pixel) -> torch.Tensor:
        return _como_tensor(self.modelo.get_image_features(**entradas_pixel))

    def features_texto(self, **tok) -> torch.Tensor:
        return _como_tensor(self.modelo.get_text_features(**tok))

    # -- API de alto nível: retorna np.ndarray (N, D) normalizado ------------
    @torch.no_grad()
    def codificar_imagens(
        self,
        imagens: Sequence[Image.Image | str | Path],
        tamanho_lote: int = config.TAMANHO_LOTE_INFERENCIA,
        mostrar_progresso: bool = False,
    ) -> np.ndarray:
        lotes = list(_em_lotes(list(imagens), tamanho_lote))
        if mostrar_progresso:
            from tqdm import tqdm

            lotes = tqdm(lotes, desc="Vetorizando imagens", unit="lote")
        saida = []
        for lote in lotes:
            pil = [carregar_imagem(im) if isinstance(im, (str, Path)) else im.convert("RGB") for im in lote]
            saida.append(normalizar_l2(self.features_imagem(**self.valores_pixel(pil))).cpu())
        return torch.cat(saida).numpy().astype(np.float32)

    @torch.no_grad()
    def codificar_textos(
        self,
        textos: Sequence[str],
        tamanho_lote: int = 128,
        mostrar_progresso: bool = False,
    ) -> np.ndarray:
        lotes = list(_em_lotes(list(textos), tamanho_lote))
        if mostrar_progresso:
            from tqdm import tqdm

            lotes = tqdm(lotes, desc="Vetorizando textos", unit="lote")
        saida = [normalizar_l2(self.features_texto(**self.tokenizar(b))).cpu() for b in lotes]
        return torch.cat(saida).numpy().astype(np.float32)

    def codificar_texto(self, texto: str) -> np.ndarray:
        return self.codificar_textos([texto])[0]

    def codificar_imagem(self, imagem: Image.Image | str | Path) -> np.ndarray:
        return self.codificar_imagens([imagem])[0]

    @torch.no_grad()
    def zero_shot(self, imagem: Image.Image, rotulos: Sequence[str], modelo_frase: str = "uma foto de {}") -> dict[str, float]:
        """Classificação zero-shot: probabilidade RELATIVA entre os rótulos, softmax(τ · I·Tᵀ).

        O SigLIP foi treinado com sigmoide (cada par avaliado isoladamente), mas com rótulos
        curtos a sigmoide fica perto de 0 para todos, mesmo o correto. O softmax sobre os
        mesmos logits ordena bem os rótulos e é o que faz sentido mostrar ao usuário.
        """
        img = torch.from_numpy(self.codificar_imagem(imagem))
        txt = torch.from_numpy(self.codificar_textos([modelo_frase.format(l) for l in rotulos]))
        logits = self.modelo.logit_scale.exp().cpu() * (txt @ img)
        return dict(zip(rotulos, logits.softmax(dim=0).tolist()))
