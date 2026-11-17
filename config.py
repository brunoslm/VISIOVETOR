"""Configurações globais do VisioVetor."""

import os
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
DIR_RAIZ = Path(__file__).resolve().parent
DIR_DADOS = DIR_RAIZ / "data"
DIR_BRUTOS = DIR_DADOS / "raw"
DIR_IMAGENS = DIR_BRUTOS / "images"
ARQUIVO_LEGENDAS = DIR_BRUTOS / "captions.jsonl"  # 1 linha por imagem: image, captions, tags, split
DIR_PROCESSADOS = DIR_DADOS / "processed"
DIR_EMBEDDINGS = DIR_DADOS / "embeddings"
DIR_MODELOS = DIR_RAIZ / "models"

# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
# SigLIP 2 é multilíngue e entende português muito melhor que o CLIP original da OpenAI
# (validação Flickr8k PT-BR, 1000 imagens, texto→imagem R@1: CLIP B/32 10,8% | SigLIP 2 base 66,2%).
# Qualquer modelo da família CLIP/SigLIP do Hub funciona, ex.: "openai/clip-vit-base-patch32".
MODELO_BASE = os.getenv("VISIOVETOR_BASE_MODEL", "google/siglip2-base-patch16-224")
DIR_MODELO_AJUSTADO = DIR_MODELOS / "ptbr-finetuned"
# Se a pasta acima não existir (ex.: num Space do Hugging Face), baixa o modelo ajustado deste repositório do Hub.
# Vazio = só usa a pasta local.
MODELO_AJUSTADO_HUB = os.getenv("VISIOVETOR_FT_MODEL", "")

# ---------------------------------------------------------------------------
# Dataset (Hugging Face) — Flickr8k com legendas traduzidas para português
# ---------------------------------------------------------------------------
DATASET_HF = os.getenv("VISIOVETOR_HF_DATASET", "laicsiifes/flickr8k-pt-br")
N_IMAGENS_TREINO = int(os.getenv("VISIOVETOR_N_TRAIN", "6000"))
N_IMAGENS_VAL = int(os.getenv("VISIOVETOR_N_VAL", "1000"))

# ---------------------------------------------------------------------------
# Hardware / inferência
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DISPOSITIVO = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DISPOSITIVO = "mps"
else:
    DISPOSITIVO = "cpu"

TAMANHO_LOTE_INFERENCIA = 32
SEMENTE = 42

# ---------------------------------------------------------------------------
# Tags determinísticas (filtro híbrido). Derivadas das legendas por palavra-chave.
# ---------------------------------------------------------------------------
PALAVRAS_CHAVE_TAGS: dict[str, list[str]] = {
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
