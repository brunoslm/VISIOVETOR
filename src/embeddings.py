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


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """v̂ = v / ||v||₂ (por linha)."""
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _batched(items: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _as_tensor(out) -> torch.Tensor:
    """get_*_features devolve tensor (transformers 4.x) ou um objeto com pooler_output (5.x)."""
    return out if torch.is_tensor(out) else out.pooler_output


class ClipEncoder:
    """Encoder de imagens e textos para modelos da família CLIP/SigLIP."""

    def __init__(self, model_path: str | Path = config.BASE_MODEL, device: str = config.DEVICE):
        self.model_path = str(model_path)
        self.device = device
        self.model = AutoModel.from_pretrained(self.model_path).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        # SigLIP foi treinado com texto em minúsculas, padding fixo de 64 e perda sigmoide
        self.is_siglip = "siglip" in self.model.config.model_type

    @property
    def embed_dim(self) -> int:
        return int(self.encode_texts(["teste"]).shape[1])

    # -- tokenização / features brutas (usadas também no fine-tuning) -------
    def tokenize(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        if self.is_siglip:
            texts = [t.lower() for t in texts]
            tok = self.tokenizer(list(texts), padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        else:
            tok = self.tokenizer(list(texts), padding=True, max_length=77, truncation=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in tok.items() if k in ("input_ids", "attention_mask")}

    def pixel_values(self, images: Sequence[Image.Image]) -> dict[str, torch.Tensor]:
        inputs = self.processor(images=list(images), return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items() if k != "input_ids"}

    def image_features(self, **pixel_inputs) -> torch.Tensor:
        return _as_tensor(self.model.get_image_features(**pixel_inputs))

    def text_features(self, **tok) -> torch.Tensor:
        return _as_tensor(self.model.get_text_features(**tok))

    # -- API de alto nível: retorna np.ndarray (N, D) normalizado ------------
    @torch.no_grad()
    def encode_images(
        self,
        images: Sequence[Image.Image | str | Path],
        batch_size: int = config.INFERENCE_BATCH_SIZE,
        show_progress: bool = False,
    ) -> np.ndarray:
        batches = list(_batched(list(images), batch_size))
        if show_progress:
            from tqdm import tqdm

            batches = tqdm(batches, desc="Vetorizando imagens", unit="lote")
        out = []
        for batch in batches:
            pil = [load_image(im) if isinstance(im, (str, Path)) else im.convert("RGB") for im in batch]
            out.append(l2_normalize(self.image_features(**self.pixel_values(pil))).cpu())
        return torch.cat(out).numpy().astype(np.float32)

    @torch.no_grad()
    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 128,
        show_progress: bool = False,
    ) -> np.ndarray:
        batches = list(_batched(list(texts), batch_size))
        if show_progress:
            from tqdm import tqdm

            batches = tqdm(batches, desc="Vetorizando textos", unit="lote")
        out = [l2_normalize(self.text_features(**self.tokenize(b))).cpu() for b in batches]
        return torch.cat(out).numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]

    def encode_image(self, image: Image.Image | str | Path) -> np.ndarray:
        return self.encode_images([image])[0]

    @torch.no_grad()
    def zero_shot(self, image: Image.Image, labels: Sequence[str], template: str = "uma foto de {}") -> dict[str, float]:
        """Classificação zero-shot.

        CLIP: softmax(τ · I·Tᵀ) — os rótulos competem entre si.
        SigLIP: sigmoid(τ · I·Tᵀ + b) — cada rótulo é avaliado de forma independente.
        """
        img = torch.from_numpy(self.encode_image(image))
        txt = torch.from_numpy(self.encode_texts([template.format(l) for l in labels]))
        logits = self.model.logit_scale.exp().cpu() * (txt @ img)
        if self.is_siglip:
            probs = torch.sigmoid(logits + self.model.logit_bias.cpu()).tolist()
        else:
            probs = logits.softmax(dim=0).tolist()
        return dict(zip(labels, probs))
