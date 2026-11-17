"""Construção/carregamento dos índices vetoriais (imagens e legendas) por modelo."""

from __future__ import annotations

import os
import re
from pathlib import Path

import config
from src.data import carregar_registros
from src.embeddings import CodificadorClip
from src.vector_store import IndiceVetorial


def baixar_modelo_ajustado() -> None:
    """Sem a pasta local do modelo ajustado, baixa do Hub (VISIOVETOR_FT_MODEL) para o mesmo lugar."""
    if (config.DIR_MODELO_AJUSTADO / "config.json").exists() or not config.MODELO_AJUSTADO_HUB:
        return
    from huggingface_hub import snapshot_download

    print(f"⬇️  Baixando o modelo ajustado de {config.MODELO_AJUSTADO_HUB}...")
    snapshot_download(config.MODELO_AJUSTADO_HUB, local_dir=config.DIR_MODELO_AJUSTADO)
    # Os pesos baixados são os mesmos usados para gerar os índices publicados junto com o app.
    # Sem isso, a data do download faria construir_indices achar o índice velho e recalcular tudo.
    for pesos in config.DIR_MODELO_AJUSTADO.glob("*.safetensors"):
        os.utime(pesos, (0, 0))


def modelos_disponiveis() -> dict[str, str]:
    """Rótulo amigável -> caminho/ID do modelo."""
    baixar_modelo_ajustado()
    modelos = {f"Base · {config.MODELO_BASE}": config.MODELO_BASE}
    if (config.DIR_MODELO_AJUSTADO / "config.json").exists():
        modelos[f"Ajustado em PT-BR · {config.DIR_MODELO_AJUSTADO.relative_to(config.DIR_RAIZ)}"] = str(config.DIR_MODELO_AJUSTADO)
    return modelos


def _slug(caminho_modelo: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", Path(caminho_modelo).name if Path(caminho_modelo).exists() else caminho_modelo).strip("_")


def prefixo_indice(caminho_modelo: str, tipo: str) -> Path:
    return config.DIR_EMBEDDINGS / f"{_slug(caminho_modelo)}__{tipo}"


def construir_indices(codificador: CodificadorClip, forcar: bool = False) -> tuple[IndiceVetorial, IndiceVetorial]:
    """Retorna (índice de imagens, índice de legendas), usando cache em data/embeddings."""
    prefixo_img = prefixo_indice(codificador.caminho_modelo, "images")
    prefixo_leg = prefixo_indice(codificador.caminho_modelo, "captions")
    registros = carregar_registros()

    if not forcar and IndiceVetorial.existe(prefixo_img) and IndiceVetorial.existe(prefixo_leg):
        imagens, legendas = IndiceVetorial.carregar(prefixo_img), IndiceVetorial.carregar(prefixo_leg)
        # o índice guarda o caminho absoluto da máquina onde foi gerado: refaz a partir da pasta atual
        for m in imagens.metadados:
            m["path"] = str(config.DIR_BRUTOS / m["image"])
        # modelo local re-treinado depois do índice => índice velho
        pesos = list(Path(codificador.caminho_modelo).glob("*.safetensors")) if Path(codificador.caminho_modelo).is_dir() else []
        hora_indice = prefixo_img.with_suffix(".npy").stat().st_mtime
        modelo_desatualizado = any(w.stat().st_mtime > hora_indice for w in pesos)
        if len(imagens) == len(registros) and not modelo_desatualizado:
            return imagens, legendas
        print("⚠️  Catálogo ou modelo mudou desde o último índice — reconstruindo.")

    print(f"🔧 Construindo índices para {codificador.caminho_modelo} ({len(registros)} imagens)...")
    meta_img = [
        {"image": r["image"], "path": r["path"], "caption": r["captions"][0], "tags": r["tags"], "split": r["split"]}
        for r in registros
    ]
    imagens = IndiceVetorial()
    imagens.adicionar(codificador.codificar_imagens([r["path"] for r in registros], mostrar_progresso=True), meta_img)

    meta_leg = [
        {"caption": c, "image": r["image"], "tags": r["tags"]}
        for r in registros
        for c in r["captions"]
    ]
    legendas = IndiceVetorial()
    legendas.adicionar(codificador.codificar_textos([m["caption"] for m in meta_leg], mostrar_progresso=True), meta_leg)

    imagens.salvar(prefixo_img)
    legendas.salvar(prefixo_leg)
    print(f"✅ Índices salvos em {config.DIR_EMBEDDINGS}")
    return imagens, legendas
