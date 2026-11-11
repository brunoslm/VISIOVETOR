"""Configurações globais do VisioVetor."""

import os
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = RAW_DIR / "images"
CAPTIONS_FILE = RAW_DIR / "captions.jsonl"  # 1 linha por imagem: image, captions, tags, split
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_DIR = ROOT_DIR / "models"

# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
# SigLIP 2 é multilíngue e entende português muito melhor que o CLIP original da OpenAI
# (validação Flickr8k PT-BR, texto→imagem R@1: CLIP B/32 19,7% | SigLIP 2 base 75,5%).
# Qualquer modelo da família CLIP/SigLIP do Hub funciona, ex.: "openai/clip-vit-base-patch32".
BASE_MODEL = os.getenv("VISIOVETOR_BASE_MODEL", "google/siglip2-base-patch16-224")
FINETUNED_MODEL_DIR = MODELS_DIR / "ptbr-finetuned"

# ---------------------------------------------------------------------------
# Dataset (Hugging Face) — Flickr8k com legendas traduzidas para português
# ---------------------------------------------------------------------------
HF_DATASET = os.getenv("VISIOVETOR_HF_DATASET", "laicsiifes/flickr8k-pt-br")
N_TRAIN_IMAGES = int(os.getenv("VISIOVETOR_N_TRAIN", "2000"))
N_VAL_IMAGES = int(os.getenv("VISIOVETOR_N_VAL", "300"))

# ---------------------------------------------------------------------------
# Hardware / inferência
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

INFERENCE_BATCH_SIZE = 32
SEED = 42

# ---------------------------------------------------------------------------
# Tags determinísticas (filtro híbrido). Derivadas das legendas por palavra-chave.
# ---------------------------------------------------------------------------
TAG_KEYWORDS: dict[str, list[str]] = {
    "cachorro": ["cachorro", "cão", "cães", "cachorros", "filhote"],
    "criança": ["criança", "crianças", "menino", "menina", "meninos", "meninas", "bebê"],
    "água": ["água", "piscina", "lago", "rio", "mar", "oceano", "onda", "ondas"],
    "praia": ["praia", "areia"],
    "neve": ["neve", "nevado", "esqui", "esquiador", "snowboard"],
    "esporte": ["futebol", "basquete", "bola", "corrida", "corre", "correndo", "bicicleta",
                "skate", "surf", "surfista", "jogador", "jogadores", "salta", "pulando"],
    "veículo": ["bicicleta", "moto", "motocicleta", "carro", "ônibus", "caminhão", "barco"],
    "cidade": ["rua", "calçada", "cidade", "prédio", "edifício", "multidão"],
    "natureza": ["grama", "campo", "floresta", "montanha", "árvore", "árvores", "rocha", "rochas"],
}
