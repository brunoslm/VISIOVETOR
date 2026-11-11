#!/usr/bin/env bash
# VisioVetor — setup + execução local (Linux/macOS)
# Variáveis opcionais: SKIP_FINETUNE=1 para pular o fine-tuning.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x venv/bin/python ]; then
  echo "🐍 Criando ambiente virtual..."
  python3 -m venv venv
fi
PY=venv/bin/python

if ! $PY -c "import torch, transformers, gradio, datasets" 2>/dev/null; then
  echo "📦 Instalando dependências..."
  $PY -m pip install --upgrade pip
  # sem GPU NVIDIA -> wheel CPU do PyTorch (bem menor)
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    $PY -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
  $PY -m pip install -r requirements.txt
fi

if [ ! -f data/raw/captions.jsonl ]; then
  echo "⬇️  Baixando dataset (Flickr8k PT-BR, subconjunto)..."
  $PY scripts/prepare_data.py
fi

if [ ! -f models/ptbr-finetuned/config.json ] && [ "${SKIP_FINETUNE:-0}" != "1" ]; then
  echo "🎯 Fine-tuning (alguns minutos em CPU)..."
  $PY scripts/finetune.py
fi

$PY scripts/build_index.py
echo "🚀 Abrindo interface em http://127.0.0.1:7860"
$PY app.py
