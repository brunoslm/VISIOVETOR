"""VisioVetor — interface Gradio para busca multimodal imagem <-> texto com CLIP/SigLIP."""

from __future__ import annotations

import json
from functools import lru_cache

import gradio as gr

import config
from src.catalog import available_models, build_indexes
from src.embeddings import ClipEncoder
from src.ui_theme import (
    FOOTER_HTML, THEME_JS, build_theme, hero_html, load_css, metrics_html, section_html,
)

MODELS = available_models()
DEFAULT_MODEL = list(MODELS)[-1]  # fine-tuned, se existir

EXAMPLES = [
    "um cachorro correndo na grama",
    "crianças brincando na água",
    "uma pessoa andando de bicicleta",
    "um homem escalando uma rocha",
    "pessoas na neve",
]


@lru_cache(maxsize=4)
def get_engine(label: str):
    encoder = ClipEncoder(MODELS[label])
    images, captions = build_indexes(encoder)
    return encoder, images, captions


def load_metrics() -> dict | None:
    path = config.FINETUNED_MODEL_DIR / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _gallery(results) -> list[tuple[str, str]]:
    return [(r.metadata["path"], f"{r.score:.3f} · {r.metadata['caption']}") for r in results]


# -- handlers ---------------------------------------------------------------
def text_to_image(query: str, model: str, top_k: int, tags: list[str] | None):
    if not query or not query.strip():
        raise gr.Error("Digite uma descrição para buscar.")
    enc, images, _ = get_engine(model)
    results = images.search(enc.encode_text(query.strip()), top_k=int(top_k), tags=tags or None)
    if not results:
        gr.Warning("Nenhuma imagem tem todas essas tags.")
    return _gallery(results)


def image_to_image(image, model: str, top_k: int, tags: list[str] | None):
    if image is None:
        raise gr.Error("Envie uma imagem.")
    enc, images, _ = get_engine(model)
    return _gallery(images.search(enc.encode_image(image), top_k=int(top_k), tags=tags or None))


def image_to_text(image, model: str, labels: str, top_k: int):
    if image is None:
        raise gr.Error("Envie uma imagem.")
    enc, _, captions = get_engine(model)
    vec = enc.encode_image(image)

    # 1) legendas mais próximas do catálogo (recuperação imagem -> texto)
    seen, rows = set(), []
    for r in captions.search(vec, top_k=int(top_k) * 3):
        cap = r.metadata["caption"]
        if cap.lower() not in seen:
            seen.add(cap.lower())
            rows.append([round(r.score, 3), cap])
        if len(rows) >= int(top_k):
            break

    # 2) classificação zero-shot com rótulos livres
    label_list = [l.strip() for l in labels.split(",") if l.strip()]
    zero_shot = enc.zero_shot(image, label_list) if len(label_list) >= 2 else {}
    return rows, zero_shot


def compare_models(query: str, top_k: int):
    if len(MODELS) < 2:
        raise gr.Error("Treine o modelo primeiro: python scripts/finetune.py")
    labels = list(MODELS)
    return text_to_image(query, labels[0], top_k, None), text_to_image(query, labels[1], top_k, None)


# -- UI -----------------------------------------------------------------------
def _gallery_component(label: str, columns: int = 4) -> gr.Gallery:
    return gr.Gallery(
        label=label, show_label=False, columns=columns, height="auto",
        object_fit="cover", allow_preview=True, elem_classes="vv-gallery",
    )


def build_ui() -> gr.Blocks:
    enc, images, captions = get_engine(DEFAULT_MODEL)
    tags = images.all_tags()
    metrics = load_metrics()
    gain = (metrics["best"]["t2i_R@1"] - metrics["baseline"]["t2i_R@1"]) if metrics else None

    with gr.Blocks(title="VisioVetor") as demo:
        gr.HTML(
            hero_html(len(images), len(captions), len(tags), config.DEVICE, gain,
                      dim=images.vectors.shape[1], model_name=config.BASE_MODEL),
            js_on_load=THEME_JS,
        )

        with gr.Row(elem_classes="vv-toolbar"):
            model = gr.Dropdown(
                list(MODELS), value=DEFAULT_MODEL, label="Modelo",
                info="O índice de cada modelo fica em cache; a primeira troca leva alguns segundos.",
            )

        with gr.Tabs(elem_classes="vv-tabs"):
            # ---------------------------------------------------------------- texto -> imagem
            with gr.Tab("Texto → Imagem"):
                gr.HTML(section_html("sky", "Busca por descrição", "Descreva a cena em português",
                                     "O texto vira um vetor e é comparado com todas as imagens do catálogo."))
                with gr.Row(equal_height=True):
                    q = gr.Textbox(show_label=False, placeholder="Ex.: um cachorro correndo na grama",
                                   scale=5, container=False, autofocus=True)
                    btn = gr.Button("Buscar", variant="primary", scale=1, min_width=110)
                with gr.Row(elem_classes="vv-chips"):
                    chips = [gr.Button(e, size="sm", variant="secondary", elem_classes="vv-chip") for e in EXAMPLES]
                with gr.Row():
                    k1 = gr.Slider(1, 32, value=8, step=1, label="Resultados")
                    t1 = gr.Dropdown(tags, multiselect=True, label="Filtrar por tags",
                                     info="A imagem precisa ter todas as tags escolhidas.")
                g1 = _gallery_component("Resultados")
                for ev in (btn.click, q.submit):
                    ev(text_to_image, [q, model, k1, t1], g1)
                for chip, text in zip(chips, EXAMPLES):
                    chip.click(lambda t=text: t, None, q).then(text_to_image, [q, model, k1, t1], g1)

            # ---------------------------------------------------------------- imagem -> imagem
            with gr.Tab("Imagem → Imagem"):
                gr.HTML(section_html("violet", "Similaridade visual", "Encontre imagens parecidas",
                                     "Envie uma foto e veja as mais próximas no espaço vetorial."))
                with gr.Row():
                    im2 = gr.Image(type="pil", label="Imagem de consulta", height=320, scale=2)
                    with gr.Column(scale=1, min_width=240):
                        k2 = gr.Slider(1, 32, value=8, step=1, label="Resultados")
                        t2 = gr.Dropdown(tags, multiselect=True, label="Filtrar por tags")
                        b2 = gr.Button("Buscar parecidas", variant="primary")
                g2 = _gallery_component("Imagens similares")
                b2.click(image_to_image, [im2, model, k2, t2], g2)

            # ---------------------------------------------------------------- imagem -> texto
            with gr.Tab("Imagem → Texto"):
                gr.HTML(section_html("teal", "Descrição automática", "Do pixel para o texto",
                                     "Recupera as legendas mais próximas do catálogo e classifica a imagem "
                                     "com rótulos livres, sem treino extra (zero-shot)."))
                with gr.Row():
                    im3 = gr.Image(type="pil", label="Imagem", height=320, scale=2)
                    with gr.Column(scale=1, min_width=240):
                        labels = gr.Textbox(
                            value="cachorro, gato, criança, praia, neve, bicicleta, futebol, montanha",
                            label="Rótulos", info="Separados por vírgula.", lines=2,
                        )
                        k3 = gr.Slider(1, 15, value=5, step=1, label="Legendas")
                        b3 = gr.Button("Descrever", variant="primary")
                with gr.Row():
                    caps = gr.Dataframe(headers=["Score", "Legenda mais próxima"], label="Legendas do catálogo",
                                        column_widths=["90px", "auto"], wrap=True, interactive=False, scale=3)
                    zs = gr.Label(label="Classificação zero-shot", num_top_classes=5, scale=2)
                b3.click(image_to_text, [im3, model, labels, k3], [caps, zs])

            # ---------------------------------------------------------------- comparação
            with gr.Tab("Base × Ajustado"):
                gr.HTML(section_html("amber", "Fine-tuning", "Quanto o ajuste em português ajudou",
                                     "Métricas medidas só nas imagens de validação, que não entram no treino."))
                gr.HTML(metrics_html(metrics))
                with gr.Row(equal_height=True, elem_classes="vv-spaced"):
                    q4 = gr.Textbox(value=EXAMPLES[0], show_label=False, placeholder="Descreva uma cena",
                                    scale=5, container=False)
                    b4 = gr.Button("Comparar", variant="primary", scale=1, min_width=110)
                k4 = gr.Slider(1, 12, value=4, step=1, label="Resultados por modelo")
                with gr.Row():
                    with gr.Column():
                        gr.HTML('<p class="vv-eyebrow"><span class="vv-dot vv-zinc"></span>Modelo base</p>')
                        g4a = _gallery_component("Base", columns=2)
                    with gr.Column():
                        gr.HTML('<p class="vv-eyebrow"><span class="vv-dot vv-sky"></span>Ajustado em português</p>')
                        g4b = _gallery_component("Ajustado", columns=2)
                for ev in (b4.click, q4.submit):
                    ev(compare_models, [q4, k4], [g4a, g4b])

        gr.HTML(FOOTER_HTML)
    return demo


if __name__ == "__main__":
    build_ui().launch(theme=build_theme(), css=load_css())
