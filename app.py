"""VisioVetor — interface Gradio para busca multimodal imagem <-> texto com CLIP/SigLIP."""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr

import config
from src.catalog import modelos_disponiveis, construir_indices
from src.embeddings import CodificadorClip
from src import jeffrey
from src.data import carregar_registros
from src.jeffrey_chat import SUGESTOES, JeffreyChat
from src.jeffrey_llm import JeffreyLLM, estatisticas_score
from src.persona import carregar_fatos
from src.ui_theme import (
    CABECALHO_CHAT, JS_RODAPE, html_intro_chat, JS_TEMA, html_rodape, construir_tema, html_destaque, carregar_css, html_metricas, html_secao,
)

MODELOS = modelos_disponiveis()
MODELO_PADRAO = list(MODELOS)[-1]  # fine-tuned, se existir

EXEMPLOS_BUSCA = [
    "um cachorro correndo na grama",
    "crianças brincando na água",
    "uma pessoa andando de bicicleta",
    "um homem escalando uma rocha",
    "pessoas na neve",
]


@lru_cache(maxsize=4)
def obter_motor(rotulo: str):
    codificador = CodificadorClip(MODELOS[rotulo])
    imagens, legendas = construir_indices(codificador)
    return codificador, imagens, legendas


@lru_cache(maxsize=512)
def _vetor_texto(modelo: str, consulta: str):
    """Vetor da consulta em cache: repetir a busca ou mudar só tags/Top-K não reprocessa o texto."""
    vet = obter_motor(modelo)[0].codificar_texto(consulta)
    vet.setflags(write=False)
    return vet


def imagens_exemplo(n: int = 6) -> list[str]:
    """Imagens de validação (fora do treino) com tags variadas, para testar as abas de imagem sem upload."""
    escolhidas, vistos = [], set()
    for r in carregar_registros(divisao="val"):
        chave = tuple(r["tags"][:1])
        if chave and chave not in vistos and Path(r["path"]).exists():
            vistos.add(chave)
            escolhidas.append(r["path"])
        if len(escolhidas) >= n:
            break
    return escolhidas


def carregar_metricas() -> dict | None:
    caminho = config.DIR_MODELO_AJUSTADO / "metrics.json"
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else None


def _galeria(resultados) -> list[tuple[str, str]]:
    return [(r.metadados["path"], f"{r.score:.3f} · {r.metadados['caption']}") for r in resultados]


def _contexto(tipo: str, resultados, consulta: str | None = None, passos=None, **extra) -> dict:
    """Contexto da última busca de uma aba, para o Jeffrey comentar no chat."""
    return {"kind": tipo, "query": consulta, "steps": list(passos or []), **extra,
            "results": [{"score": float(r.score), "caption": r.metadados["caption"], "tags": r.metadados.get("tags", [])}
                        for r in resultados]}


def _triplas(resultados) -> list[tuple[float, str, list[str]]]:
    return [(r.score, r.metadados["caption"], r.metadados.get("tags", [])) for r in resultados]


def _nome_modelo(rotulo: str) -> str:
    """Nome curto do modelo para o Jeffrey citar."""
    return "SigLIP 2 ajustado em português" if rotulo != list(MODELOS)[0] else MODELOS[rotulo]


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


# -- handlers ---------------------------------------------------------------
def texto_para_imagem(consulta: str, modelo: str, top_k: int, tags: list[str] | None):
    if not consulta or not consulta.strip():
        raise gr.Error("Digite uma descrição para buscar.")
    consulta = consulta.strip()
    t0 = time.perf_counter()
    cod, imagens, _ = obter_motor(modelo)
    resultados = imagens.buscar(_vetor_texto(modelo, consulta), top_k=int(top_k), tags=tags or None)
    nar = jeffrey.explicar_busca_texto(
        consulta, _triplas(resultados), nome_modelo=_nome_modelo(modelo), dim=imagens.vetores.shape[1],
        n_catalogo=len(imagens), tags=tags or None, n_apos_filtro=imagens.contar_com_tags(tags), ms=_ms(t0),
    )
    return _galeria(resultados), nar.html(), _contexto("texto", resultados, consulta, nar.passos, tags=tags or None)


def imagem_para_imagem(imagem, modelo: str, top_k: int, tags: list[str] | None):
    if imagem is None:
        raise gr.Error("Envie uma imagem.")
    t0 = time.perf_counter()
    cod, imagens, _ = obter_motor(modelo)
    resultados = imagens.buscar(cod.codificar_imagem(imagem), top_k=int(top_k), tags=tags or None)
    nar = jeffrey.explicar_busca_imagem(
        _triplas(resultados), nome_modelo=_nome_modelo(modelo), dim=imagens.vetores.shape[1], n_catalogo=len(imagens),
        tags=tags or None, n_apos_filtro=imagens.contar_com_tags(tags), ms=_ms(t0),
    )
    return _galeria(resultados), nar.html(), _contexto("imagem", resultados, None, nar.passos, tags=tags or None)


def imagem_para_texto(imagem, modelo: str, rotulos: str, top_k: int):
    if imagem is None:
        raise gr.Error("Envie uma imagem.")
    t0 = time.perf_counter()
    cod, _, legendas = obter_motor(modelo)
    vet = cod.codificar_imagem(imagem)

    # 1) legendas mais próximas do catálogo (recuperação imagem -> texto)
    vistos, linhas = set(), []
    for r in legendas.buscar(vet, top_k=int(top_k) * 3):
        leg = r.metadados["caption"]
        if leg.lower() not in vistos:
            vistos.add(leg.lower())
            linhas.append([round(r.score, 3), leg])
        if len(linhas) >= int(top_k):
            break

    # 2) classificação zero-shot com rótulos livres
    lista_rotulos = [l.strip() for l in rotulos.split(",") if l.strip()]
    zero_shot = cod.zero_shot(imagem, lista_rotulos) if len(lista_rotulos) >= 2 else {}
    nar = jeffrey.explicar_imagem_para_texto(
        [(r[0], r[1]) for r in linhas], zero_shot, e_siglip=cod.e_siglip, nome_modelo=_nome_modelo(modelo), ms=_ms(t0),
    )
    ctx = {"kind": "descrever", "query": None, "steps": nar.passos, "zero_shot": zero_shot,
           "results": [{"score": float(sc), "caption": c, "tags": []} for sc, c in linhas]}
    return linhas, zero_shot, nar.html(), ctx


def comparar_modelos(consulta: str, top_k: int):
    if len(MODELOS) < 2:
        raise gr.Error("Treine o modelo primeiro: python scripts/finetune.py")
    if not consulta or not consulta.strip():
        raise gr.Error("Digite uma descrição para comparar.")
    consulta = consulta.strip()
    saida = []
    for rotulo in list(MODELOS)[:2]:
        imagens = obter_motor(rotulo)[1]
        saida.append(imagens.buscar(_vetor_texto(rotulo, consulta), top_k=int(top_k)))
    base, ajustado = ([(r.score, r.metadados["caption"], r.metadados["image"]) for r in res] for res in saida)
    nar = jeffrey.explicar_comparacao(consulta, base, ajustado)
    ctx = _contexto("comparar", saida[1], consulta, nar.passos,
               base=[{"score": float(r.score), "caption": r.metadados["caption"]} for r in saida[0]])
    return _galeria(saida[0]), _galeria(saida[1]), nar.html(), ctx


# -- chat com o Jeffrey --------------------------------------------------------
JEFF = JeffreyChat(carregar_registros(), fatos=carregar_fatos())
LLM: JeffreyLLM | None = None  # criado em construir_interface, carrega em segundo plano
PENSANDO = '<span class="vv-typing" aria-label="Jeffrey está digitando"><i></i><i></i><i></i></span>'
JEFF_AVATAR = str(Path(__file__).with_name("src") / "assets" / "jeffrey.svg")


def _turno_chat(hist: list, msg: str, resp: str) -> list:
    return (hist or []) + [{"role": "user", "content": msg}, {"role": "assistant", "content": resp}]


def _fluxo_jeffrey(msg: str, hist: list, ctx: dict | None, tipo: str):
    """Texto parcial da resposta. LLM quando disponível; regras como plano B."""
    if (a := JEFF.responder_seguranca(msg)):
        yield a
    elif (a := JEFF.analise(msg, ctx)):
        yield a  # pergunta específica (posição, "qual tem X", catálogo): resposta exata, sem LLM
    elif LLM is not None and LLM.disponivel:
        yield from LLM.gerar_em_fluxo(msg, ctx, hist, tipo)
    else:
        yield JEFF.responder(msg, ctx, tipo)[0]


def _turno_em_fluxo(msg: str, hist: list, ctx: dict | None, tipo: str):
    hist = list(hist or []) + [{"role": "user", "content": msg}]
    yield hist + [{"role": "assistant", "content": PENSANDO}]
    for parcial in _fluxo_jeffrey(msg, hist[:-1], ctx, tipo):
        yield hist + [{"role": "assistant", "content": parcial or PENSANDO}]


def funcao_chat(tipo: str):
    """Conversa sobre a última busca da aba. O chat não roda buscas: elas ficam na caixa do topo."""
    def fn(msg, hist, ctx):
        msg = (msg or "").strip()
        if not msg:
            yield gr.skip(), gr.skip()
            return
        if JEFF.pedido_de_busca(msg):
            yield "", gr.update(value=_turno_chat(hist, msg, JEFF.responder(msg, ctx, tipo)[0]), visible=True)
            return
        for h in _turno_em_fluxo(msg, hist, ctx, tipo):
            yield "", gr.update(value=h, visible=True)
    return fn


# -- UI -----------------------------------------------------------------------
def _componente_galeria(rotulo: str, colunas: int = 4) -> gr.Gallery:
    return gr.Gallery(
        label=rotulo, show_label=False, columns=colunas, height="auto",
        object_fit="cover", allow_preview=True, elem_classes="vv-gallery",
    )


def _revelar(fn):
    """Roda a busca e mostra a área de resultados (escondida até a primeira busca)."""
    def envolvida(*args):
        return (*fn(*args), gr.update(visible=True))
    return envolvida


def bloco_chat(tipo: str):
    """Chat com o Jeffrey logo abaixo da fala dele. Devolve (chips, chatbot, caixa, botão)."""
    gr.HTML(html_intro_chat(tipo), elem_classes="vv-chat-intro")
    with gr.Group(elem_classes="vv-chat"):
        gr.HTML(CABECALHO_CHAT)
        chatbot = gr.Chatbot(show_label=False, visible=False, height=340, layout="bubble",
                         elem_classes="vv-chat-log", buttons=[], avatar_images=(None, JEFF_AVATAR))
        with gr.Row(equal_height=True, elem_classes="vv-chat-input"):
            caixa = gr.Textbox(show_label=False, container=False, scale=5, elem_classes="vv-chat-box",
                             placeholder="Pergunte ao Jeffrey sobre os resultados")
            enviar = gr.Button("Enviar", variant="primary", scale=0, min_width=96, elem_classes="vv-chat-send")
    with gr.Row(elem_classes="vv-chips vv-chat-chips"):
        chips = [gr.Button(s, size="sm", variant="secondary", elem_classes="vv-chip") for s in SUGESTOES[tipo]]
    return chips, chatbot, caixa, enviar


def _ligar_chat(chips, caixa, enviar, fn, entradas, saidas):
    for evento in (enviar.click, caixa.submit):
        evento(fn, entradas, saidas)
    for chip in chips:
        chip.click(lambda t=chip.value: t, None, caixa).then(fn, entradas, saidas)


def construir_interface() -> gr.Blocks:
    global LLM
    cod, imagens, legendas = obter_motor(MODELO_PADRAO)
    tags = imagens.todas_tags()
    LLM = JeffreyLLM(JEFF, carregar_fatos(), estatisticas_score(imagens, legendas))
    LLM.carregar_em_segundo_plano()
    metricas = carregar_metricas()
    ganho = (metricas["best"]["t2i_R@1"] - metricas["baseline"]["t2i_R@1"]) if metricas else None
    amostras = imagens_exemplo()

    with gr.Blocks(title="VisioVetor") as interface:
        gr.HTML(
            html_destaque(len(imagens), len(legendas), len(tags), config.DISPOSITIVO, ganho,
                      dim=imagens.vetores.shape[1], nome_modelo=config.MODELO_BASE),
            js_on_load=JS_TEMA,
        )

        with gr.Row(elem_classes="vv-toolbar"):
            modelo = gr.Dropdown(
                list(MODELOS), value=MODELO_PADRAO, label="Modelo",
                info="O índice de cada modelo fica em cache; a primeira troca leva alguns segundos.",
            )

        with gr.Tabs(elem_classes="vv-tabs"):
            # ---------------------------------------------------------------- texto -> imagem
            with gr.Tab("Texto → Imagem"):
                gr.HTML(html_secao("sky", "Busca por descrição", "Descreva a cena em português",
                                     "O texto vira um vetor e é comparado com todas as imagens do catálogo."))
                with gr.Row(equal_height=True):
                    q = gr.Textbox(show_label=False, placeholder="Ex.: um cachorro correndo na grama",
                                   scale=5, container=False)
                    botao = gr.Button("Buscar", variant="primary", scale=1, min_width=110)
                with gr.Row(elem_classes="vv-chips"):
                    chips = [gr.Button(e, size="sm", variant="secondary", elem_classes="vv-chip") for e in EXEMPLOS_BUSCA]
                with gr.Row():
                    k1 = gr.Slider(1, 32, value=8, step=1, label="Resultados")
                    t1 = gr.Dropdown(tags, multiselect=True, label="Filtrar por tags",
                                     info="A imagem precisa ter todas as tags escolhidas.")
                with gr.Column(visible=False) as r1:
                    g1 = _componente_galeria("Resultados")
                j1 = gr.HTML(jeffrey.boas_vindas("texto").html())
                ctx1 = gr.State(None)
                c1chips, c1bot, c1box, c1send = bloco_chat("texto")
                for evento in (botao.click, q.submit):
                    evento(_revelar(texto_para_imagem), [q, modelo, k1, t1], [g1, j1, ctx1, r1])
                for chip, texto in zip(chips, EXEMPLOS_BUSCA):
                    chip.click(lambda t=texto: t, None, q).then(_revelar(texto_para_imagem), [q, modelo, k1, t1], [g1, j1, ctx1, r1])
                _ligar_chat(c1chips, c1box, c1send, funcao_chat("texto"), [c1box, c1bot, ctx1], [c1box, c1bot])

            # ---------------------------------------------------------------- imagem -> imagem
            with gr.Tab("Imagem → Imagem"):
                gr.HTML(html_secao("violet", "Similaridade visual", "Encontre imagens parecidas",
                                     "Envie uma foto e veja as mais próximas no espaço vetorial."))
                with gr.Row():
                    im2 = gr.Image(type="pil", label="Imagem de consulta", height=320, scale=2)
                    with gr.Column(scale=1, min_width=240):
                        k2 = gr.Slider(1, 32, value=8, step=1, label="Resultados")
                        t2 = gr.Dropdown(tags, multiselect=True, label="Filtrar por tags")
                        b2 = gr.Button("Buscar parecidas", variant="primary")
                if amostras:
                    gr.Examples(amostras, inputs=im2, label="Sem foto à mão? Clique num exemplo",
                                examples_per_page=len(amostras))
                with gr.Column(visible=False) as r2:
                    g2 = _componente_galeria("Imagens similares")
                j2 = gr.HTML(jeffrey.boas_vindas("imagem").html())
                ctx2 = gr.State(None)
                c2chips, c2bot, c2box, c2send = bloco_chat("imagem")
                b2.click(_revelar(imagem_para_imagem), [im2, modelo, k2, t2], [g2, j2, ctx2, r2])
                _ligar_chat(c2chips, c2box, c2send, funcao_chat("imagem"), [c2box, c2bot, ctx2], [c2box, c2bot])

            # ---------------------------------------------------------------- imagem -> texto
            with gr.Tab("Imagem → Texto"):
                gr.HTML(html_secao("teal", "Descrição automática", "Do pixel para o texto",
                                     "Recupera as legendas mais próximas do catálogo e classifica a imagem "
                                     "com rótulos livres, sem treino extra (zero-shot)."))
                with gr.Row():
                    im3 = gr.Image(type="pil", label="Imagem", height=320, scale=2)
                    with gr.Column(scale=1, min_width=240):
                        rotulos = gr.Textbox(
                            value="cachorro, gato, criança, praia, neve, bicicleta, futebol, montanha",
                            label="Rótulos", info="Separados por vírgula.", lines=2,
                        )
                        k3 = gr.Slider(1, 15, value=5, step=1, label="Legendas")
                        b3 = gr.Button("Descrever", variant="primary")
                if amostras:
                    gr.Examples(amostras, inputs=im3, label="Sem foto à mão? Clique num exemplo",
                                examples_per_page=len(amostras))
                with gr.Row(visible=False) as r3:
                    tabela_legendas = gr.Dataframe(headers=["Score", "Legenda mais próxima"], label="Legendas do catálogo",
                                        column_widths=["90px", "auto"], wrap=True, interactive=False, scale=3)
                    zs = gr.Label(label="Classificação zero-shot", num_top_classes=5, scale=2)
                j3 = gr.HTML(jeffrey.boas_vindas("descrever").html())
                ctx3 = gr.State(None)
                c3chips, c3bot, c3box, c3send = bloco_chat("descrever")
                b3.click(_revelar(imagem_para_texto), [im3, modelo, rotulos, k3], [tabela_legendas, zs, j3, ctx3, r3])
                _ligar_chat(c3chips, c3box, c3send, funcao_chat("descrever"), [c3box, c3bot, ctx3], [c3box, c3bot])

            # ---------------------------------------------------------------- comparação
            with gr.Tab("Base × Ajustado"):
                gr.HTML(html_secao("amber", "Fine-tuning", "Quanto o ajuste em português ajudou",
                                     "Métricas medidas só nas imagens de validação, que não entram no treino."))
                gr.HTML(html_metricas(metricas))
                with gr.Row(equal_height=True, elem_classes="vv-spaced"):
                    q4 = gr.Textbox(value=EXEMPLOS_BUSCA[0], show_label=False, placeholder="Descreva uma cena",
                                    scale=5, container=False)
                    b4 = gr.Button("Comparar", variant="primary", scale=1, min_width=110)
                k4 = gr.Slider(1, 12, value=4, step=1, label="Resultados por modelo")
                with gr.Row(visible=False) as r4:
                    with gr.Column():
                        gr.HTML('<p class="vv-eyebrow"><span class="vv-dot vv-zinc"></span>Modelo base</p>')
                        g4a = _componente_galeria("Base", colunas=2)
                    with gr.Column():
                        gr.HTML('<p class="vv-eyebrow"><span class="vv-dot vv-sky"></span>Ajustado em português</p>')
                        g4b = _componente_galeria("Ajustado", colunas=2)
                j4 = gr.HTML(jeffrey.boas_vindas("comparar").html())
                ctx4 = gr.State(None)
                c4chips, c4bot, c4box, c4send = bloco_chat("comparar")
                for evento in (b4.click, q4.submit):
                    evento(_revelar(comparar_modelos), [q4, k4], [g4a, g4b, j4, ctx4, r4])
                _ligar_chat(c4chips, c4box, c4send, funcao_chat("comparar"), [c4box, c4bot, ctx4], [c4box, c4bot])

        gr.HTML(html_rodape(config.MODELO_BASE, imagens.vetores.shape[1], config.DISPOSITIVO, len(imagens)),
                js_on_load=JS_RODAPE)
    return interface


if __name__ == "__main__":
    construir_interface().launch(theme=construir_tema(), css=carregar_css())
