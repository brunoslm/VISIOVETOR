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

OPENERS_EXPLICA = ["Claro, vamos olhar juntos.", "Com prazer!", "Boa busca!", "Vamos lá.", "Claro!"]
PEDIDOS = ["Pode me explicar esses resultados?", "O que você acha desses resultados?", "Esses resultados fazem sentido?",
           "Me ajuda a interpretar isso?", "O que esses scores querem dizer?", "Isso ficou bom?"]
PEDIDOS_PASSOS = ["O que você fez nessa busca?", "Como você chegou nesses resultados?",
                  "Me explica passo a passo o que aconteceu.", "Como funcionou essa busca?"]


@dataclass
class Index:
    img_vecs: np.ndarray
    img_meta: list[dict]
    cap_vecs: np.ndarray
    cap_meta: list[dict]

    @property
    def image_pos(self) -> dict[str, int]:
        return {m["image"]: i for i, m in enumerate(self.img_meta)}


def load_index(slug: str) -> Index | None:
    p = config.EMBEDDINGS_DIR / slug
    files = [Path(f"{p}__{k}.{e}") for k in ("images", "captions") for e in ("npy", "json")]
    if not all(f.exists() for f in files):
        return None
    return Index(
        img_vecs=np.load(files[0]), img_meta=json.loads(files[1].read_text(encoding="utf-8")),
        cap_vecs=np.load(files[2]), cap_meta=json.loads(files[3].read_text(encoding="utf-8")),
    )


def _q(caption: str) -> str:
    """Legenda -> texto de busca (minúscula no início, sem ponto final)."""
    c = caption.strip().rstrip(".").strip()
    return c[:1].lower() + c[1:]


def _results_block(res: list[tuple[float, str, list[str]]], label: str = "score") -> str:
    return "\n".join(
        f"{k}. {label} {fmt(s, 3)} — \"{c.strip().rstrip('.')}\"" + (f" [tags: {', '.join(t)}]" if t else "")
        for k, (s, c, t) in enumerate(res, 1)
    )


def _val_queries(idx: Index, rng: random.Random, n: int, min_w: int = 4, max_w: int = 14) -> list[int]:
    val_images = {m["image"] for m in idx.img_meta if m.get("split") == "val"}
    cands = [i for i, m in enumerate(idx.cap_meta)
             if m["image"] in val_images and min_w <= len(m["caption"].split()) <= max_w]
    return rng.sample(cands, min(n, len(cands)))


def _top(idx: Index, vec: np.ndarray, k: int, exclude: set[int] | None = None) -> list[int]:
    scores = idx.img_vecs @ vec
    order = np.argsort(-scores)
    return [int(i) for i in order if not exclude or int(i) not in exclude][:k]


# ---------------------------------------------------------------------------
# Texto -> Imagem: explicação de resultados (e "o que eu fiz")
# ---------------------------------------------------------------------------
def explicacoes(idx: Index, n: int, rng: random.Random, model_name: str) -> list[dict]:
    out = []
    for ci in _val_queries(idx, rng, n):
        query = _q(idx.cap_meta[ci]["caption"])
        scores = idx.img_vecs @ idx.cap_vecs[ci]
        k = rng.choice([3, 4, 5, 5, 6])
        top = _top(idx, idx.cap_vecs[ci], k)
        res = [(float(scores[i]), idx.img_meta[i]["caption"], idx.img_meta[i].get("tags", [])) for i in top]
        com_passos = rng.random() < 0.35
        pedido = rng.choice(PEDIDOS_PASSOS if com_passos else PEDIDOS)
        user = f"Busquei por \"{query}\" e estes foram os {k} primeiros resultados:\n{_results_block(res)}\n\n{pedido}"
        nar = jeffrey.explain_text_search(query, res, model_name=model_name, dim=F.dim, n_catalog=F.n_images,
                                          openers=OPENERS_EXPLICA)
        out.append(chat(user, nar.text(with_steps=com_passos)))
    return out


# ---------------------------------------------------------------------------
# Imagem -> Imagem
# ---------------------------------------------------------------------------
def imagem_para_imagem(idx: Index, n: int, rng: random.Random, model_name: str) -> list[dict]:
    val = [i for i, m in enumerate(idx.img_meta) if m.get("split") == "val"]
    out = []
    for qi in rng.sample(val, min(n, len(val))):
        k = rng.choice([4, 5, 6])
        top = _top(idx, idx.img_vecs[qi], k, exclude={qi})  # simula uma foto nova: tira a própria imagem
        scores = idx.img_vecs @ idx.img_vecs[qi]
        res = [(float(scores[i]), idx.img_meta[i]["caption"], idx.img_meta[i].get("tags", [])) for i in top]
        foto = _q(idx.img_meta[qi]["caption"])
        user = (f"Enviei uma foto ({foto}) na busca por imagens parecidas e o sistema trouxe:\n"
                f"{_results_block(res, 'similaridade')}\n\n{rng.choice(PEDIDOS)}")
        nar = jeffrey.explain_image_search(res, model_name=model_name, dim=F.dim, n_catalog=F.n_images,
                                           openers=OPENERS_EXPLICA)
        out.append(chat(user, nar.text(with_steps=False)))
    return out


# ---------------------------------------------------------------------------
# Imagem -> Texto (legendas mais próximas + zero-shot)
# ---------------------------------------------------------------------------
LABEL_POOL = ["cachorro", "gato", "criança", "homem", "mulher", "praia", "neve", "água", "grama", "bicicleta",
              "futebol", "montanha", "cidade", "rua", "carro", "barco", "cavalo", "piscina", "bola", "floresta",
              "rocha", "multidão", "skate", "surf", "pássaro"]


def imagem_para_texto(idx: Index, n: int, rng: random.Random, label_vecs: np.ndarray | None,
                      logit_scale: float | None) -> list[dict]:
    val = [i for i, m in enumerate(idx.img_meta) if m.get("split") == "val"]
    own = {}
    for j, m in enumerate(idx.cap_meta):
        own.setdefault(m["image"], set()).add(j)
    out = []
    for qi in rng.sample(val, min(n, len(val))):
        image = idx.img_meta[qi]["image"]
        cs = idx.cap_vecs @ idx.img_vecs[qi]
        rows, seen = [], set()
        for j in np.argsort(-cs):
            j = int(j)
            if j in own[image]:
                continue  # simula uma foto nova: ignora as legendas da própria imagem
            c = idx.cap_meta[j]["caption"].strip()
            if c.lower() not in seen:
                seen.add(c.lower())
                rows.append((float(cs[j]), c))
            if len(rows) >= 3:
                break

        zs = None
        if label_vecs is not None and logit_scale:
            # rótulos relevantes (palavras das legendas da própria foto) + distratores aleatórios
            own_words = {w for j in own[image] for w in re.findall(r"\w+", idx.cap_meta[j]["caption"].lower())}
            rel = [i for i, l in enumerate(LABEL_POOL) if l in own_words]
            rng.shuffle(rel)
            rel = rel[:rng.choice([1, 2])]
            others = [i for i in range(len(LABEL_POOL)) if i not in rel]
            lab_idx = rel + rng.sample(others, rng.choice([3, 4, 5]))
            rng.shuffle(lab_idx)
            logits = logit_scale * (label_vecs[lab_idx] @ idx.img_vecs[qi])
            p = np.exp(logits - logits.max())
            p /= p.sum()
            zs = {LABEL_POOL[i]: float(v) for i, v in zip(lab_idx, p)}

        linhas = "\n".join(f"{k}. score {fmt(s, 3)} — \"{c.rstrip('.')}\"" for k, (s, c) in enumerate(rows, 1))
        user = f"Mandei uma foto na aba Imagem → Texto. Legendas mais próximas:\n{linhas}"
        if zs:
            ranked = sorted(zs.items(), key=lambda kv: -kv[1])
            user += "\nZero-shot: " + ", ".join(f"{k} {fmt(100 * v, 0)}%" for k, v in ranked)
        user += "\n\n" + rng.choice(["O que isso diz sobre a minha foto?", "Pode explicar?", "Como eu leio isso?"])
        nar = jeffrey.explain_image_to_text(rows, zs, is_siglip=True)
        out.append(chat(user, nar.text(with_steps=False)))
    return out


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


def refinamento(idx: Index, rng: random.Random, per_tag: int = 6) -> list[dict]:
    counts = Counter(t for m in idx.img_meta for t in m.get("tags", []))
    out = []
    for tag, sins in TAG_SINONIMOS.items():
        if tag not in counts:
            continue
        for _ in range(per_tag):
            v = rng.choice(sins)
            # sugestões = legendas reais que contêm a MESMA palavra que o usuário usou
            pat = re.compile(rf"\b{re.escape(v)}\b", re.IGNORECASE)
            pool = sorted({m["caption"] for m in idx.cap_meta
                           if pat.search(m["caption"]) and 5 <= len(m["caption"].split()) <= 10})
            if len(pool) < 3:
                continue
            sug = rng.sample(pool, 3)
            user = rng.choice(PEDIDO_VAGO).format(v=v)
            ideias = "\n".join(f"- \"{_q(c)}\"" for c in sug)
            resp = (f"{rng.choice(['Com prazer!', 'Claro!', 'Vamos lá!'])} O catálogo tem {fmt(counts[tag], 0)} "
                    f"imagens com a tag \"{tag}\", então vale descrever a cena para chegar no que você imagina. "
                    f"Algumas ideias tiradas do próprio catálogo:\n{ideias}\n"
                    f"Também dá para marcar a tag \"{tag}\" no filtro para ver só essas imagens.")
            out.append(chat(user, resp))
    return out


# ---------------------------------------------------------------------------
# Estatísticas do catálogo (honestidade: "tem foto de X?")
# ---------------------------------------------------------------------------
PALAVRAS = ["gato", "cavalo", "girafa", "elefante", "avião", "bolo", "guitarra", "pássaro", "barco", "trem",
            "moto", "chapéu", "balão", "piscina", "skate", "surf", "tênis", "basquete", "futebol", "casamento",
            "montanha", "flor", "árvore", "ponte", "igreja", "ônibus", "palhaço", "fantasia", "bebê", "polícia",
            "bicicleta", "cachorro", "praia", "neve", "computador", "pizza", "vaca", "ovelha", "leão", "baleia",
            "violão", "escola", "hospital", "fogueira", "pipa", "patins", "caiaque", "rodeio", "dança", "corda"]


def _count_word(idx: Index, word: str) -> tuple[int, list[str]]:
    """Quantas imagens têm a palavra em alguma das legendas, e exemplos de legendas."""
    pat = re.compile(rf"\b{re.escape(word)}s?\b", re.IGNORECASE)
    hits = [m for m in idx.cap_meta if pat.search(m["caption"])]
    return len({m["image"] for m in hits}), [m["caption"] for m in hits[:5]]


def catalogo(idx: Index, rng: random.Random) -> list[dict]:
    out = []
    tag_counts = Counter(t for m in idx.img_meta for t in m.get("tags", []))
    total = len(idx.img_meta)

    for tag, n in sorted(tag_counts.items()):
        pct = fmt(100 * n / total, 0)
        for pergunta in (f"Quantas fotos de {tag} tem?", f"Tem muita imagem com {tag}?"):
            out.append(chat(pergunta,
                            f"Tem {fmt(n, 0)} imagens com a tag \"{tag}\", cerca de {pct}% das {N_IMG} do catálogo. "
                            "As tags são atribuídas automaticamente a partir das legendas, então pode haver fotos "
                            f"de {tag} sem a tag; a busca por texto encontra essas também."))

    lista = "\n".join(f"- {t}: {fmt(n, 0)} imagens" for t, n in tag_counts.most_common())
    for pergunta in ("Quais tags existem?", "Me lista as tags", "Quais filtros eu posso usar?"):
        out.append(chat(pergunta, f"Com prazer! Hoje existem {len(tag_counts)} tags, criadas automaticamente a "
                                  f"partir das legendas:\n{lista}\nQuando você escolhe mais de uma, a imagem precisa "
                                  "ter todas."))
    mais, menos = tag_counts.most_common()[0], tag_counts.most_common()[-1]
    out.append(chat("Qual a tag mais comum?", f"É \"{mais[0]}\", com {fmt(mais[1], 0)} imagens. Faz sentido: o "
                                             "Flickr8k tem muitas fotos desse tipo."))
    out.append(chat("Qual a tag mais rara?", f"É \"{menos[0]}\", com {fmt(menos[1], 0)} imagens. Para esse tema, "
                                            "as buscas têm menos opções para escolher."))

    for w in PALAVRAS:
        n, exemplos = _count_word(idx, w)
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
        out.append(chat(pergunta, resp))
    return out


# ---------------------------------------------------------------------------
# Base x Ajustado com resultados reais dos dois índices
# ---------------------------------------------------------------------------
def comparacao(base: Index, tuned: Index, n: int, rng: random.Random) -> list[dict]:
    out = []
    for ci in _val_queries(tuned, rng, n):
        query = _q(tuned.cap_meta[ci]["caption"])
        k = rng.choice([3, 4, 5])
        lists = []
        for idx in (base, tuned):
            vec = idx.cap_vecs[ci]  # mesma ordem de legendas nos dois índices
            sc = idx.img_vecs @ vec
            lists.append([(float(sc[i]), idx.img_meta[i]["caption"], idx.img_meta[i]["image"]) for i in _top(idx, vec, k)])
        b, t = lists
        bloco = (f"Modelo base:\n" + "\n".join(f"{j}. \"{c.rstrip('.')}\"" for j, (_, c, _) in enumerate(b, 1)) +
                 f"\nModelo ajustado:\n" + "\n".join(f"{j}. \"{c.rstrip('.')}\"" for j, (_, c, _) in enumerate(t, 1)))
        user = f"Comparei a busca \"{query}\" nos dois modelos:\n{bloco}\n\nO que mudou?"
        out.append(chat(user, jeffrey.explain_compare(query, b, t).text(with_steps=False)))
    return out


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


def conversas_geradas(idx: Index, n: int, rng: random.Random, model_name: str) -> list[dict]:
    out = []
    for ci in _val_queries(idx, rng, n):
        query = _q(idx.cap_meta[ci]["caption"])
        sc = idx.img_vecs @ idx.cap_vecs[ci]
        top = _top(idx, idx.cap_vecs[ci], 4)
        res = [(float(sc[i]), idx.img_meta[i]["caption"], idx.img_meta[i].get("tags", [])) for i in top]
        s_u, s_a = rng.choice(SAUDACOES)
        d_u, d_a = rng.choice(DESPEDIDAS)
        nar = jeffrey.explain_text_search(query, res, model_name=model_name, dim=F.dim, n_catalog=F.n_images,
                                          openers=OPENERS_EXPLICA)
        turns = [s_u, s_a,
                 f"Busquei \"{query}\" e veio isto:\n{_results_block(res)}\n\n{rng.choice(PEDIDOS)}", nar.text()]
        if rng.random() < 0.6:
            # refinamento baseado nas legendas reais dos resultados
            ideias = "\n".join(f"- \"{_q(c)}\"" for _, c, _ in res[:3])
            turns += [rng.choice(["Como deixo a busca mais certeira?", "E se eu quiser algo mais específico?"]),
                      "Uma boa estratégia é copiar o estilo das legendas que chegaram perto do que você quer, "
                      f"acrescentando o detalhe que faltou. Por exemplo:\n{ideias}\n"
                      "Escolha a mais próxima da sua ideia e ajuste cor, lugar ou ação."]
        turns += [d_u, d_a]
        out.append(chat(*turns))
    return out


def label_vectors(model_path: str) -> tuple[np.ndarray | None, float | None]:
    """Vetores dos rótulos do zero-shot (carrega o modelo só para o texto)."""
    try:
        from src.embeddings import ClipEncoder

        enc = ClipEncoder(model_path)
        vecs = enc.encode_texts([f"uma foto de {l}" for l in LABEL_POOL])
        return vecs, float(enc.model.logit_scale.exp().item())
    except Exception as e:  # sem modelo local: gera imagem_para_texto sem zero-shot
        print(f"⚠️  Não consegui carregar o modelo para o zero-shot ({e}); seguindo sem ele.")
        return None, None
