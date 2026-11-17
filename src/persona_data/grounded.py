"""Datasets do Jeffrey ancorados nos dados REAIS do projeto.

Tudo aqui vem dos índices de vetores (data/embeddings), das legendas e das tags do
catálogo. As explicações usam o mesmo narrador do app (src/jeffrey.py), então o
modelo aprende a falar exatamente como o painel da interface.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from src import jeffrey
from src.persona import fmt
from src.persona_data.common import F, N_IMG, chat

ABERTURAS_EXPLICA = ["Claro, vamos olhar juntos.", "Com prazer!", "Boa busca!", "Vamos lá.", "Claro!"]
PEDIDOS = ["Pode me explicar esses resultados?", "O que você acha desses resultados?", "Esses resultados fazem sentido?",
           "Me ajuda a interpretar isso?", "O que esses scores querem dizer?", "Isso ficou bom?"]
PEDIDOS_PASSOS = ["O que você fez nessa busca?", "Como você chegou nesses resultados?",
                  "Me explica passo a passo o que aconteceu.", "Como funcionou essa busca?"]


@dataclass
class Indice:
    vetores_img: np.ndarray
    meta_img: list[dict]
    vetores_leg: np.ndarray
    meta_leg: list[dict]

    @property
    def pos_imagem(self) -> dict[str, int]:
        return {m["image"]: i for i, m in enumerate(self.meta_img)}


def carregar_indice(slug: str) -> Indice | None:
    p = config.DIR_EMBEDDINGS / slug
    arquivos = [Path(f"{p}__{k}.{e}") for k in ("images", "captions") for e in ("npy", "json")]
    if not all(f.exists() for f in arquivos):
        return None
    return Indice(
        vetores_img=np.load(arquivos[0]), meta_img=json.loads(arquivos[1].read_text(encoding="utf-8")),
        vetores_leg=np.load(arquivos[2]), meta_leg=json.loads(arquivos[3].read_text(encoding="utf-8")),
    )


def _q(legenda: str) -> str:
    """Legenda -> texto de busca (minúscula no início, sem ponto final)."""
    c = legenda.strip().rstrip(".").strip()
    return c[:1].lower() + c[1:]


def _bloco_resultados(res: list[tuple[float, str, list[str]]], rotulo: str = "score") -> str:
    return "\n".join(
        f"{k}. {rotulo} {fmt(s, 3)} — \"{c.strip().rstrip('.')}\"" + (f" [tags: {', '.join(t)}]" if t else "")
        for k, (s, c, t) in enumerate(res, 1)
    )


def _consultas_val(idx: Indice, rng: random.Random, n: int, min_pal: int = 4, max_pal: int = 14) -> list[int]:
    imagens_val = {m["image"] for m in idx.meta_img if m.get("split") == "val"}
    candidatas = [i for i, m in enumerate(idx.meta_leg)
             if m["image"] in imagens_val and min_pal <= len(m["caption"].split()) <= max_pal]
    return rng.sample(candidatas, min(n, len(candidatas)))


def _top(idx: Indice, vet: np.ndarray, k: int, excluir: set[int] | None = None) -> list[int]:
    scores = idx.vetores_img @ vet
    ordem = np.argsort(-scores)
    return [int(i) for i in ordem if not excluir or int(i) not in excluir][:k]


# ---------------------------------------------------------------------------
# Texto -> Imagem: explicação de resultados (e "o que eu fiz")
# ---------------------------------------------------------------------------
def explicacoes(idx: Indice, n: int, rng: random.Random, nome_modelo: str) -> list[dict]:
    saida = []
    for ci in _consultas_val(idx, rng, n):
        consulta = _q(idx.meta_leg[ci]["caption"])
        scores = idx.vetores_img @ idx.vetores_leg[ci]
        k = rng.choice([3, 4, 5, 5, 6])
        topo = _top(idx, idx.vetores_leg[ci], k)
        res = [(float(scores[i]), idx.meta_img[i]["caption"], idx.meta_img[i].get("tags", [])) for i in topo]
        com_passos = rng.random() < 0.35
        pedido = rng.choice(PEDIDOS_PASSOS if com_passos else PEDIDOS)
        usuario = f"Busquei por \"{consulta}\" e estes foram os {k} primeiros resultados:\n{_bloco_resultados(res)}\n\n{pedido}"
        nar = jeffrey.explicar_busca_texto(consulta, res, nome_modelo=nome_modelo, dim=F.dim, n_catalogo=F.n_imagens,
                                          aberturas=ABERTURAS_EXPLICA)
        saida.append(chat(usuario, nar.texto(com_passos=com_passos)))
    return saida


# ---------------------------------------------------------------------------
# Imagem -> Imagem
# ---------------------------------------------------------------------------
def imagem_para_imagem(idx: Indice, n: int, rng: random.Random, nome_modelo: str) -> list[dict]:
    val = [i for i, m in enumerate(idx.meta_img) if m.get("split") == "val"]
    saida = []
    for qi in rng.sample(val, min(n, len(val))):
        k = rng.choice([4, 5, 6])
        topo = _top(idx, idx.vetores_img[qi], k, excluir={qi})  # simula uma foto nova: tira a própria imagem
        scores = idx.vetores_img @ idx.vetores_img[qi]
        res = [(float(scores[i]), idx.meta_img[i]["caption"], idx.meta_img[i].get("tags", [])) for i in topo]
        foto = _q(idx.meta_img[qi]["caption"])
        usuario = (f"Enviei uma foto ({foto}) na busca por imagens parecidas e o sistema trouxe:\n"
                f"{_bloco_resultados(res, 'similaridade')}\n\n{rng.choice(PEDIDOS)}")
        nar = jeffrey.explicar_busca_imagem(res, nome_modelo=nome_modelo, dim=F.dim, n_catalogo=F.n_imagens,
                                           aberturas=ABERTURAS_EXPLICA)
        saida.append(chat(usuario, nar.texto(com_passos=False)))
    return saida


# ---------------------------------------------------------------------------
# Imagem -> Texto (legendas mais próximas + zero-shot)
# ---------------------------------------------------------------------------
ROTULOS_ZERO_SHOT = ["cachorro", "gato", "criança", "homem", "mulher", "praia", "neve", "água", "grama", "bicicleta",
              "futebol", "montanha", "cidade", "rua", "carro", "barco", "cavalo", "piscina", "bola", "floresta",
              "rocha", "multidão", "skate", "surf", "pássaro"]


def imagem_para_texto(idx: Indice, n: int, rng: random.Random, vetores_rotulos: np.ndarray | None,
                      escala_logit: float | None) -> list[dict]:
    val = [i for i, m in enumerate(idx.meta_img) if m.get("split") == "val"]
    proprias = {}
    for j, m in enumerate(idx.meta_leg):
        proprias.setdefault(m["image"], set()).add(j)
    saida = []
    for qi in rng.sample(val, min(n, len(val))):
        imagem = idx.meta_img[qi]["image"]
        cs = idx.vetores_leg @ idx.vetores_img[qi]
        pares_leg, vistos = [], set()
        for j in np.argsort(-cs):
            j = int(j)
            if j in proprias[imagem]:
                continue  # simula uma foto nova: ignora as legendas da própria imagem
            c = idx.meta_leg[j]["caption"].strip()
            if c.lower() not in vistos:
                vistos.add(c.lower())
                pares_leg.append((float(cs[j]), c))
            if len(pares_leg) >= 3:
                break

        zs = None
        if vetores_rotulos is not None and escala_logit:
            # rótulos relevantes (palavras das legendas da própria foto) + distratores aleatórios
            palavras_proprias = {w for j in proprias[imagem] for w in re.findall(r"\w+", idx.meta_leg[j]["caption"].lower())}
            relevantes = [i for i, l in enumerate(ROTULOS_ZERO_SHOT) if l in palavras_proprias]
            rng.shuffle(relevantes)
            relevantes = relevantes[:rng.choice([1, 2])]
            outros = [i for i in range(len(ROTULOS_ZERO_SHOT)) if i not in relevantes]
            idx_rotulos = relevantes + rng.sample(outros, rng.choice([3, 4, 5]))
            rng.shuffle(idx_rotulos)
            logits = escala_logit * (vetores_rotulos[idx_rotulos] @ idx.vetores_img[qi])
            p = np.exp(logits - logits.max())
            p /= p.sum()
            zs = {ROTULOS_ZERO_SHOT[i]: float(v) for i, v in zip(idx_rotulos, p)}

        linhas = "\n".join(f"{k}. score {fmt(s, 3)} — \"{c.rstrip('.')}\"" for k, (s, c) in enumerate(pares_leg, 1))
        usuario = f"Mandei uma foto na aba Imagem → Texto. Legendas mais próximas:\n{linhas}"
        if zs:
            ordenados = sorted(zs.items(), key=lambda kv: -kv[1])
            usuario += "\nZero-shot: " + ", ".join(f"{k} {fmt(100 * v, 0)}%" for k, v in ordenados)
        usuario += "\n\n" + rng.choice(["O que isso diz sobre a minha foto?", "Pode explicar?", "Como eu leio isso?"])
        nar = jeffrey.explicar_imagem_para_texto(pares_leg, zs, e_siglip=True)
        saida.append(chat(usuario, nar.texto(com_passos=False)))
    return saida


# ---------------------------------------------------------------------------
# Refinamento de busca vaga -> sugestões tiradas do próprio catálogo
# ---------------------------------------------------------------------------
TAG_SINONIMOS = {
    "cachorro": ["cachorro", "cachorros", "cães", "cachorrinho"],
    "criança": ["criança", "crianças", "menino", "menina"],
    "água": ["água", "piscina", "lago", "rio"],
    "praia": ["praia", "areia", "mar"],
    "neve": ["neve", "inverno", "esqui"],
    "esporte": ["esporte", "esportes", "futebol", "corrida"],
    "veículo": ["veículo", "carro", "bicicleta", "barco"],
    "cidade": ["cidade", "rua", "urbano"],
    "natureza": ["natureza", "montanha", "floresta", "campo"],
}
PEDIDO_VAGO = ["Quero fotos de {v}", "busca {v}", "{v}", "Me mostra {v}", "Tem fotos de {v}?", "Procura {v} pra mim"]


def refinamento(idx: Indice, rng: random.Random, por_tag: int = 6) -> list[dict]:
    contagens = Counter(t for m in idx.meta_img for t in m.get("tags", []))
    saida = []
    for tag, sinonimos in TAG_SINONIMOS.items():
        if tag not in contagens:
            continue
        for _ in range(por_tag):
            v = rng.choice(sinonimos)
            # sugestões = legendas reais que contêm a MESMA palavra que o usuário usou
            padrao = re.compile(rf"\b{re.escape(v)}\b", re.IGNORECASE)
            opcoes = sorted({m["caption"] for m in idx.meta_leg
                           if padrao.search(m["caption"]) and 5 <= len(m["caption"].split()) <= 10})
            if len(opcoes) < 3:
                continue
            sug = rng.sample(opcoes, 3)
            usuario = rng.choice(PEDIDO_VAGO).format(v=v)
            ideias = "\n".join(f"- \"{_q(c)}\"" for c in sug)
            resp = (f"{rng.choice(['Com prazer!', 'Claro!', 'Vamos lá!'])} O catálogo tem {fmt(contagens[tag], 0)} "
                    f"imagens com a tag \"{tag}\", então vale descrever a cena para chegar no que você imagina. "
                    f"Algumas ideias tiradas do próprio catálogo:\n{ideias}\n"
                    f"Também dá para marcar a tag \"{tag}\" no filtro para ver só essas imagens.")
            saida.append(chat(usuario, resp))
    return saida


# ---------------------------------------------------------------------------
# Estatísticas do catálogo (honestidade: "tem foto de X?")
# ---------------------------------------------------------------------------
PALAVRAS = ["gato", "cavalo", "girafa", "elefante", "avião", "bolo", "guitarra", "pássaro", "barco", "trem",
            "moto", "chapéu", "balão", "piscina", "skate", "surf", "tênis", "basquete", "futebol", "casamento",
            "montanha", "flor", "árvore", "ponte", "igreja", "ônibus", "palhaço", "fantasia", "bebê", "polícia",
            "bicicleta", "cachorro", "praia", "neve", "computador", "pizza", "vaca", "ovelha", "leão", "baleia",
            "violão", "escola", "hospital", "fogueira", "pipa", "patins", "caiaque", "rodeio", "dança", "corda"]


def _contar_palavra(idx: Indice, palavra: str) -> tuple[int, list[str]]:
    """Quantas imagens têm a palavra em alguma das legendas, e exemplos de legendas."""
    padrao = re.compile(rf"\b{re.escape(palavra)}s?\b", re.IGNORECASE)
    achados = [m for m in idx.meta_leg if padrao.search(m["caption"])]
    return len({m["image"] for m in achados}), [m["caption"] for m in achados[:5]]


def catalogo(idx: Indice, rng: random.Random) -> list[dict]:
    saida = []
    contagem_tags = Counter(t for m in idx.meta_img for t in m.get("tags", []))
    total = len(idx.meta_img)

    for tag, n in sorted(contagem_tags.items()):
        pct = fmt(100 * n / total, 0)
        for pergunta in (f"Quantas fotos de {tag} tem?", f"Tem muita imagem com {tag}?"):
            saida.append(chat(pergunta,
                            f"Tem {fmt(n, 0)} imagens com a tag \"{tag}\", cerca de {pct}% das {N_IMG} do catálogo. "
                            "As tags são atribuídas automaticamente a partir das legendas, então pode haver fotos "
                            f"de {tag} sem a tag; a busca por texto encontra essas também."))

    lista = "\n".join(f"- {t}: {fmt(n, 0)} imagens" for t, n in contagem_tags.most_common())
    for pergunta in ("Quais tags existem?", "Me lista as tags", "Quais filtros eu posso usar?"):
        saida.append(chat(pergunta, f"Com prazer! Hoje existem {len(contagem_tags)} tags, criadas automaticamente a "
                                  f"partir das legendas:\n{lista}\nQuando você escolhe mais de uma, a imagem precisa "
                                  "ter todas."))
    mais, menos = contagem_tags.most_common()[0], contagem_tags.most_common()[-1]
    saida.append(chat("Qual a tag mais comum?", f"É \"{mais[0]}\", com {fmt(mais[1], 0)} imagens. Faz sentido: o "
                                             "Flickr8k tem muitas fotos desse tipo."))
    saida.append(chat("Qual a tag mais rara?", f"É \"{menos[0]}\", com {fmt(menos[1], 0)} imagens. Para esse tema, "
                                            "as buscas têm menos opções para escolher."))

    for w in PALAVRAS:
        n, exemplos = _contar_palavra(idx, w)
        pergunta = rng.choice([f"Tem foto de {w}?", f"Existe alguma imagem com {w}?", f"Vocês têm {w} no catálogo?"])
        if n == 0:
            resp = (f"Vou ser sincero: não encontrei nenhuma legenda com \"{w}\" no catálogo. Posso mostrar o que "
                    f"existe de mais parecido, mas provavelmente não será {w}. Se você adicionar fotos assim, a busca "
                    "passa a encontrá-las.")
        elif n <= 3:
            resp = (f"Encontrei poucas: {n} {'imagem tem' if n == 1 else 'imagens têm'} \"{w}\" nas legendas, por "
                    f"exemplo \"{_q(exemplos[0])}\". Busque por algo parecido com essa descrição para achá-las.")
        else:
            resp = (f"Sim! {fmt(n, 0)} imagens mencionam \"{w}\" nas legendas, como \"{_q(exemplos[0])}\". Uma busca "
                    f"descrevendo a cena com {w} deve trazê-las nos primeiros resultados.")
        saida.append(chat(pergunta, resp))
    return saida


# ---------------------------------------------------------------------------
# Base x Ajustado com resultados reais dos dois índices
# ---------------------------------------------------------------------------
def comparacao(base: Indice, ajustado: Indice, n: int, rng: random.Random) -> list[dict]:
    saida = []
    for ci in _consultas_val(ajustado, rng, n):
        consulta = _q(ajustado.meta_leg[ci]["caption"])
        k = rng.choice([3, 4, 5])
        listas = []
        for idx in (base, ajustado):
            vet = idx.vetores_leg[ci]  # mesma ordem de legendas nos dois índices
            sc = idx.vetores_img @ vet
            listas.append([(float(sc[i]), idx.meta_img[i]["caption"], idx.meta_img[i]["image"]) for i in _top(idx, vet, k)])
        b, t = listas
        bloco = (f"Modelo base:\n" + "\n".join(f"{j}. \"{c.rstrip('.')}\"" for j, (_, c, _) in enumerate(b, 1)) +
                 f"\nModelo ajustado:\n" + "\n".join(f"{j}. \"{c.rstrip('.')}\"" for j, (_, c, _) in enumerate(t, 1)))
        usuario = f"Comparei a busca \"{consulta}\" nos dois modelos:\n{bloco}\n\nO que mudou?"
        saida.append(chat(usuario, jeffrey.explicar_comparacao(consulta, b, t).texto(com_passos=False)))
    return saida


# ---------------------------------------------------------------------------
# Conversas geradas: saudação -> busca explicada -> refinamento -> despedida
# ---------------------------------------------------------------------------
SAUDACOES = [("Oi Jeffrey", "Olá! Que bom ver você por aqui. Em que posso ajudar?"),
             ("Bom dia", "Bom dia! É um prazer falar com você. Em que posso ajudar hoje?"),
             ("Boa tarde, Jeffrey", "Boa tarde! Seja bem-vindo. Quer fazer uma busca?"),
             ("Olá", "Olá! Eu sou o Jeffrey, assistente do VisioVetor. Por onde quer começar?")]
DESPEDIDAS = [("Valeu, era isso!", "Por nada! Foi um prazer ajudar. Volte sempre que quiser."),
              ("Obrigado, Jeffrey", "Eu que agradeço! Boas buscas."),
              ("Perfeito, obrigada", "Fico feliz em ajudar! Até a próxima.")]


def conversas_geradas(idx: Indice, n: int, rng: random.Random, nome_modelo: str) -> list[dict]:
    saida = []
    for ci in _consultas_val(idx, rng, n):
        consulta = _q(idx.meta_leg[ci]["caption"])
        sc = idx.vetores_img @ idx.vetores_leg[ci]
        topo = _top(idx, idx.vetores_leg[ci], 4)
        res = [(float(sc[i]), idx.meta_img[i]["caption"], idx.meta_img[i].get("tags", [])) for i in topo]
        s_u, s_a = rng.choice(SAUDACOES)
        d_u, d_a = rng.choice(DESPEDIDAS)
        nar = jeffrey.explicar_busca_texto(consulta, res, nome_modelo=nome_modelo, dim=F.dim, n_catalogo=F.n_imagens,
                                          aberturas=ABERTURAS_EXPLICA)
        turnos = [s_u, s_a,
                 f"Busquei \"{consulta}\" e veio isto:\n{_bloco_resultados(res)}\n\n{rng.choice(PEDIDOS)}", nar.texto()]
        if rng.random() < 0.6:
            # refinamento baseado nas legendas reais dos resultados
            ideias = "\n".join(f"- \"{_q(c)}\"" for _, c, _ in res[:3])
            turnos += [rng.choice(["Como deixo a busca mais certeira?", "E se eu quiser algo mais específico?"]),
                      "Uma boa estratégia é copiar o estilo das legendas que chegaram perto do que você quer, "
                      f"acrescentando o detalhe que faltou. Por exemplo:\n{ideias}\n"
                      "Escolha a mais próxima da sua ideia e ajuste cor, lugar ou ação."]
        turnos += [d_u, d_a]
        saida.append(chat(*turnos))
    return saida


def calcular_vetores_rotulos(caminho_modelo: str) -> tuple[np.ndarray | None, float | None]:
    """Vetores dos rótulos do zero-shot (carrega o modelo só para o texto)."""
    try:
        from src.embeddings import CodificadorClip

        cod = CodificadorClip(caminho_modelo)
        vets = cod.codificar_textos([f"uma foto de {l}" for l in ROTULOS_ZERO_SHOT])
        return vets, float(cod.modelo.logit_scale.exp().item())
    except Exception as e:  # sem modelo local: gera imagem_para_texto sem zero-shot
        print(f"⚠️  Não consegui carregar o modelo para o zero-shot ({e}); seguindo sem ele.")
        return None, None
