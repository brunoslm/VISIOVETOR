"""VisioVetor — interface Gradio para busca multimodal imagem <-> texto com CLIP/SigLIP."""

from __future__ import annotations

import json
import time
from functools import lru_cache

import gradio as gr

import config
from src.catalog import available_models, build_indexes
from src.embeddings import ClipEncoder
from src import jeffrey
from src.data import load_records
from src.jeffrey_chat import SUGESTOES, JeffreyChat
from src.jeffrey_llm import JeffreyLLM, score_stats
from src.persona import fmt, load_facts
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


def _ctx(kind: str, results, query: str | None = None, steps=None, **extra) -> dict:
    """Contexto da última busca de uma aba, para o Jeffrey comentar no chat."""
    return {"kind": kind, "query": query, "steps": list(steps or []), **extra,
            "results": [{"score": float(r.score), "caption": r.metadata["caption"], "tags": r.metadata.get("tags", [])}
                        for r in results]}


def _triples(results) -> list[tuple[float, str, list[str]]]:
    return [(r.score, r.metadata["caption"], r.metadata.get("tags", [])) for r in results]


def _model_name(label: str) -> str:
    """Nome curto do modelo para o Jeffrey citar."""
    return "SigLIP 2 ajustado em português" if label != list(MODELS)[0] else MODELS[label]


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


# -- handlers ---------------------------------------------------------------
def text_to_image(query: str, model: str, top_k: int, tags: list[str] | None):
    if not query or not query.strip():
        raise gr.Error("Digite uma descrição para buscar.")
    query = query.strip()
    t0 = time.perf_counter()
    enc, images, _ = get_engine(model)
    results = images.search(enc.encode_text(query), top_k=int(top_k), tags=tags or None)
    nar = jeffrey.explain_text_search(
        query, _triples(results), model_name=_model_name(model), dim=images.vectors.shape[1],
        n_catalog=len(images), tags=tags or None, n_after_filter=images.count_with_tags(tags), ms=_ms(t0),
    )
    return _gallery(results), nar.html(), _ctx("texto", results, query, nar.steps, tags=tags or None)


def image_to_image(image, model: str, top_k: int, tags: list[str] | None):
    if image is None:
        raise gr.Error("Envie uma imagem.")
    t0 = time.perf_counter()
    enc, images, _ = get_engine(model)
    results = images.search(enc.encode_image(image), top_k=int(top_k), tags=tags or None)
    nar = jeffrey.explain_image_search(
        _triples(results), model_name=_model_name(model), dim=images.vectors.shape[1], n_catalog=len(images),
        tags=tags or None, n_after_filter=images.count_with_tags(tags), ms=_ms(t0),
    )
    return _gallery(results), nar.html(), _ctx("imagem", results, None, nar.steps, tags=tags or None)


def image_to_text(image, model: str, labels: str, top_k: int):
    if image is None:
        raise gr.Error("Envie uma imagem.")
    t0 = time.perf_counter()
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
    nar = jeffrey.explain_image_to_text(
        [(r[0], r[1]) for r in rows], zero_shot, is_siglip=enc.is_siglip, model_name=_model_name(model), ms=_ms(t0),
    )
    ctx = {"kind": "descrever", "query": None, "steps": nar.steps, "zero_shot": zero_shot,
           "results": [{"score": float(sc), "caption": c, "tags": []} for sc, c in rows]}
    return rows, zero_shot, nar.html(), ctx


def compare_models(query: str, top_k: int):
    if len(MODELS) < 2:
        raise gr.Error("Treine o modelo primeiro: python scripts/finetune.py")
    if not query or not query.strip():
        raise gr.Error("Digite uma descrição para comparar.")
    query = query.strip()
    out = []
    for label in list(MODELS)[:2]:
        enc, images, _ = get_engine(label)
        out.append(images.search(enc.encode_text(query), top_k=int(top_k)))
    base, tuned = ([(r.score, r.metadata["caption"], r.metadata["image"]) for r in res] for res in out)
    nar = jeffrey.explain_compare(query, base, tuned)
    ctx = _ctx("comparar", out[1], query, nar.steps,
               base=[{"score": float(r.score), "caption": r.metadata["caption"]} for r in out[0]])
    return _gallery(out[0]), _gallery(out[1]), nar.html(), ctx


# -- chat com o Jeffrey --------------------------------------------------------
JEFF = JeffreyChat(load_records(), facts=load_facts())
LLM: JeffreyLLM | None = None  # criado em build_ui, carrega em segundo plano
PENSANDO = "Pensando…"


def _chat_turn(hist: list, msg: str, ans: str) -> list:
    return (hist or []) + [{"role": "user", "content": msg}, {"role": "assistant", "content": ans}]


def _after_search(q: str, ctx: dict, verbo: str = "Busquei") -> str:
    res = ctx.get("results") or []
    if not res:
        return f"{verbo} \"{q}\", mas nenhuma imagem passou pelo filtro de tags. Que tal remover uma delas?"
    top = res[0]
    return (f"{verbo} \"{q}\" para você. O mais próximo foi \"{top['caption'].rstrip('.')}\", com score "
            f"{fmt(top['score'], 3)}. Comentei os detalhes no painel acima.")


def _jeffrey_stream(msg: str, hist: list, ctx: dict | None, kind: str):
    """Texto parcial da resposta. LLM quando disponível; regras como plano B."""
    if (a := JEFF.answer_safety(msg)):
        yield a
    elif (a := JEFF.analysis(msg, ctx)):
        yield a  # pergunta específica (posição, "qual tem X", catálogo): resposta exata, sem LLM
    elif LLM is not None and LLM.available:
        yield from LLM.stream(msg, ctx, hist, kind)
    else:
        yield JEFF.reply(msg, ctx, kind)[0]


def _stream_turn(msg: str, hist: list, ctx: dict | None, kind: str):
    hist = list(hist or []) + [{"role": "user", "content": msg}]
    yield hist + [{"role": "assistant", "content": PENSANDO}]
    for partial in _jeffrey_stream(msg, hist[:-1], ctx, kind):
        yield hist + [{"role": "assistant", "content": partial or PENSANDO}]


def chat_text(msg, hist, ctx, model, top_k, tags):
    msg = (msg or "").strip()
    if not msg:
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return
    if (q := JEFF.search_request(msg)):
        gal, html, ctx = text_to_image(q, model, top_k, tags)
        yield "", gr.update(value=_chat_turn(hist, msg, _after_search(q, ctx)), visible=True), ctx, gal, html, q
        return
    for h in _stream_turn(msg, hist, ctx, "texto"):
        yield "", gr.update(value=h, visible=True), gr.skip(), gr.skip(), gr.skip(), gr.skip()


def chat_compare(msg, hist, ctx, top_k):
    msg = (msg or "").strip()
    if not msg:
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return
    if (q := JEFF.search_request(msg)):
        ga, gb, html, ctx = compare_models(q, top_k)
        yield ("", gr.update(value=_chat_turn(hist, msg, _after_search(q, ctx, "Comparei")), visible=True), ctx,
               ga, gb, html, q)
        return
    for h in _stream_turn(msg, hist, ctx, "comparar"):
        yield "", gr.update(value=h, visible=True), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()


def chat_simple(kind: str):
    def fn(msg, hist, ctx):
        msg = (msg or "").strip()
        if not msg:
            yield gr.skip(), gr.skip()
            return
        if JEFF.search_request(msg):
            yield "", gr.update(value=_chat_turn(hist, msg, JEFF.reply(msg, ctx, kind)[0]), visible=True)
            return
        for h in _stream_turn(msg, hist, ctx, kind):
            yield "", gr.update(value=h, visible=True)
    return fn


# -- UI -----------------------------------------------------------------------
def _gallery_component(label: str, columns: int = 4) -> gr.Gallery:
    return gr.Gallery(
        label=label, show_label=False, columns=columns, height="auto",
        object_fit="cover", allow_preview=True, elem_classes="vv-gallery",
    )


def chat_block(kind: str):
    """Chat com o Jeffrey logo abaixo da fala dele. Devolve (chips, chatbot, caixa, botão)."""
    with gr.Group(elem_classes="vv-chat"):
        bot = gr.Chatbot(show_label=False, visible=False, height=300, layout="bubble",
                         elem_classes="vv-chat-log", buttons=[])
        with gr.Row(equal_height=True, elem_classes="vv-chat-input"):
            box = gr.Textbox(show_label=False, container=False, scale=5,
                             placeholder="Pergunte ao Jeffrey, ou diga \"busca\" e uma cena" if kind in ("texto", "comparar")
                             else "Pergunte ao Jeffrey sobre o resultado")
            send = gr.Button("Perguntar", variant="secondary", scale=1, min_width=110)
    with gr.Row(elem_classes="vv-chips vv-chat-chips"):
        chips = [gr.Button(s, size="sm", variant="secondary", elem_classes="vv-chip") for s in SUGESTOES[kind]]
    return chips, bot, box, send


def _wire_chat(chips, box, send, fn, inputs, outputs):
    for ev in (send.click, box.submit):
        ev(fn, inputs, outputs)
    for chip in chips:
        chip.click(lambda t=chip.value: t, None, box).then(fn, inputs, outputs)


def build_ui() -> gr.Blocks:
    global LLM
    enc, images, captions = get_engine(DEFAULT_MODEL)
    tags = images.all_tags()
    LLM = JeffreyLLM(JEFF, load_facts(), score_stats(images, captions))
    LLM.load_async()
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
                j1 = gr.HTML(jeffrey.welcome("texto").html())
                ctx1 = gr.State(None)
                c1chips, c1bot, c1box, c1send = chat_block("texto")
                g1 = _gallery_component("Resultados")
                for ev in (btn.click, q.submit):
                    ev(text_to_image, [q, model, k1, t1], [g1, j1, ctx1])
                for chip, text in zip(chips, EXAMPLES):
                    chip.click(lambda t=text: t, None, q).then(text_to_image, [q, model, k1, t1], [g1, j1, ctx1])
                _wire_chat(c1chips, c1box, c1send, chat_text, [c1box, c1bot, ctx1, model, k1, t1],
                           [c1box, c1bot, ctx1, g1, j1, q])

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
                j2 = gr.HTML(jeffrey.welcome("imagem").html())
                ctx2 = gr.State(None)
                c2chips, c2bot, c2box, c2send = chat_block("imagem")
                g2 = _gallery_component("Imagens similares")
                b2.click(image_to_image, [im2, model, k2, t2], [g2, j2, ctx2])
                _wire_chat(c2chips, c2box, c2send, chat_simple("imagem"), [c2box, c2bot, ctx2], [c2box, c2bot])

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
                j3 = gr.HTML(jeffrey.welcome("descrever").html())
                ctx3 = gr.State(None)
                c3chips, c3bot, c3box, c3send = chat_block("descrever")
                with gr.Row():
                    caps = gr.Dataframe(headers=["Score", "Legenda mais próxima"], label="Legendas do catálogo",
                                        column_widths=["90px", "auto"], wrap=True, interactive=False, scale=3)
                    zs = gr.Label(label="Classificação zero-shot", num_top_classes=5, scale=2)
                b3.click(image_to_text, [im3, model, labels, k3], [caps, zs, j3, ctx3])
                _wire_chat(c3chips, c3box, c3send, chat_simple("descrever"), [c3box, c3bot, ctx3], [c3box, c3bot])

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
                j4 = gr.HTML(jeffrey.welcome("comparar").html())
                ctx4 = gr.State(None)
                c4chips, c4bot, c4box, c4send = chat_block("comparar")
                with gr.Row():
                    with gr.Column():
                        gr.HTML('<p class="vv-eyebrow"><span class="vv-dot vv-zinc"></span>Modelo base</p>')
                        g4a = _gallery_component("Base", columns=2)
                    with gr.Column():
                        gr.HTML('<p class="vv-eyebrow"><span class="vv-dot vv-sky"></span>Ajustado em português</p>')
                        g4b = _gallery_component("Ajustado", columns=2)
                for ev in (b4.click, q4.submit):
                    ev(compare_models, [q4, k4], [g4a, g4b, j4, ctx4])
                _wire_chat(c4chips, c4box, c4send, chat_compare, [c4box, c4bot, ctx4, k4],
                           [c4box, c4bot, ctx4, g4a, g4b, j4, q4])

        gr.HTML(FOOTER_HTML)
    return demo


if __name__ == "__main__":
    build_ui().launch(theme=build_theme(), css=load_css())
