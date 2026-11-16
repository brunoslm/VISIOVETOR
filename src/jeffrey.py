"""Jeffrey narra as buscas: o que foi feito e como ler os resultados.

Usado em dois lugares, com o mesmo texto:
  * app.py  -> painel do Jeffrey acima da galeria (HTML);
  * scripts/generate_dataset.py -> respostas dos datasets de alinhamento (texto).

Tudo aqui é determinístico e baseado só nos números reais da busca: o Jeffrey
não inventa nada que o sistema não tenha calculado.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from html import escape

from src.persona import NAME, fmt

STOP = set("""a o os as um uma uns umas de da do das dos em na no nas nos num numa e é ao à com por para pela pelo
que se sua seu suas seus está estão enquanto outro outra outros outras como mais muito sobre entre até lá
ele ela eles elas isso este esta esse essa ser são tem têm foto imagem imagens fotos""".split())


def join_pt(items: list[str]) -> str:
    """["a", "b", "c"] -> 'a, b e c'"""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " e " + items[-1]


def content_words(text: str) -> list[str]:
    return list(dict.fromkeys(
        w for w in re.findall(r"\w+", text.lower()) if len(w) > 3 and w not in STOP and not w.isdigit()
    ))


def _rng(*key: object) -> random.Random:
    """Aleatoriedade estável: a mesma busca gera sempre o mesmo texto."""
    h = hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def _clean(caption: str) -> str:
    return caption.strip().rstrip(".")


@dataclass
class Narration:
    steps: list[str] = field(default_factory=list)  # "o que eu fiz"
    body: list[str] = field(default_factory=list)   # análise dos resultados
    tone: str = "info"                               # info | warn

    def text(self, with_steps: bool = False) -> str:
        parts = []
        if with_steps and self.steps:
            parts.append("O que eu fiz:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(self.steps, 1)))
        parts.append(" ".join(self.body))
        return "\n\n".join(parts)

    def html(self) -> str:
        steps = ""
        if self.steps:
            items = "".join(f"<li>{escape(s)}</li>" for s in self.steps)
            steps = (f'<details class="vv-jeffrey-steps"><summary>O que eu fiz</summary>'
                     f'<ol>{items}</ol></details>')
        body = "".join(f"<p>{escape(p)}</p>" for p in self.body)
        return (f'<div class="vv-jeffrey vv-jeffrey-{self.tone}" role="status" aria-live="polite">'
                f'<div class="vv-jeffrey-avatar" aria-hidden="true">J</div>'
                f'<div class="vv-jeffrey-content"><p class="vv-jeffrey-name">{NAME}'
                f'<span>assistente do VisioVetor</span></p>{body}{steps}</div></div>')


def welcome(kind: str = "texto") -> Narration:
    msgs = {
        "texto": "Olá! Eu sou o Jeffrey. Descreva uma cena e clique em Buscar: eu conto aqui o que fiz e como "
                 "ler os resultados.",
        "imagem": "Envie uma foto e clique em Buscar parecidas. Depois eu explico o que encontrei.",
        "descrever": "Envie uma foto e clique em Descrever. Eu mostro as legendas mais próximas e explico a "
                     "classificação zero-shot.",
        "comparar": "Digite uma busca e clique em Comparar. Eu digo o que mudou entre o modelo base e o ajustado.",
    }
    return Narration(body=[msgs[kind]])


# ---------------------------------------------------------------------------
# análises reutilizáveis
# ---------------------------------------------------------------------------
def _tempo(ms: float) -> str:
    return f"{fmt(ms / 1000)} s" if ms >= 1000 else f"{fmt(ms, 0)} ms"


def _pct(p: float) -> str:
    return "menos de 1%" if p < 0.01 else f"{fmt(100 * p, 0)}%"


def _margin_sentence(s1: float, s2: float | None) -> str:
    if s2 is None:
        return ""
    d = s1 - s2
    if d < 0.005:
        return (f"Ele está praticamente empatado com o 2º ({fmt(s2, 3)}), então a ordem entre os dois diz pouco: "
                "vale olhar ambos.")
    if d > 0.02:
        return f"Ele se destaca bem do 2º, que ficou em {fmt(s2, 3)}, o que indica uma correspondência clara."
    return f"O 2º vem logo atrás ({fmt(s2, 3)}), então os dois combinam de forma parecida com a busca."


def _terms_sentences(query: str, captions: list[str]) -> list[str]:
    qwords = content_words(query)
    caps_words = [set(content_words(c)) for c in captions]
    n = len(captions)
    presentes = [w for w in qwords if sum(w in cw for cw in caps_words) >= max(2, (n + 1) // 2)]
    ausentes = [w for w in qwords if not any(w in cw for cw in caps_words)]
    out = []
    if presentes:
        sel = presentes[:2]
        ws = " e ".join(f"\"{w}\"" for w in sel)
        verbo = "aparece" if len(sel) == 1 else "aparecem"
        artigo = "O termo" if len(sel) == 1 else "Os termos"
        out.append(f"{artigo} {ws} {verbo} na maioria das legendas dos resultados, sinal de que o modelo captou "
                   "bem essa parte da descrição.")
    if ausentes:
        out.append(f"Já \"{ausentes[0]}\" não aparece em nenhuma legenda dos resultados. As legendas não descrevem "
                   "tudo o que há na foto, mas, se esse detalhe for importante, vale destacá-lo na busca.")
    return out


def _tag_sentence(tag_lists: list[list[str]], used_tags: list[str] | None) -> str:
    counts = Counter(t for ts in tag_lists for t in ts if not used_tags or t not in used_tags)
    if not counts:
        return ""
    tag, cnt = counts.most_common(1)[0]
    n = len(tag_lists)
    if n < 3 or cnt < max(3, (n + 1) // 2):
        return ""
    onde = f"em todos os {n} resultados" if cnt == n else f"em {cnt} dos {n} resultados"
    return f"Para restringir, a tag \"{tag}\" aparece {onde}."


# ---------------------------------------------------------------------------
# Texto -> Imagem
# ---------------------------------------------------------------------------
def explain_text_search(
    query: str,
    results: list[tuple[float, str, list[str]]],
    *,
    model_name: str = "",
    dim: int | None = None,
    n_catalog: int | None = None,
    tags: list[str] | None = None,
    n_after_filter: int | None = None,
    ms: float | None = None,
    openers: list[str] | None = None,
) -> Narration:
    """results: [(score, legenda, tags)] em ordem de ranking."""
    rng = _rng("t2i", query, len(results), tags)
    nar = Narration()

    # o que eu fiz
    vec = f"um vetor de {dim} dimensões" if dim else "um vetor"
    nar.steps.append(f"Transformei \"{query}\" em {vec}" + (f" com o modelo {model_name}." if model_name else "."))
    if n_catalog:
        nar.steps.append(f"Comparei esse vetor com as {fmt(n_catalog, 0)} imagens do catálogo usando a "
                         "similaridade de cosseno.")
    if tags:
        lista = join_pt([f"\"{t}\"" for t in tags])
        sobra = f": sobraram {fmt(n_after_filter, 0)} imagens" if n_after_filter is not None else ""
        nar.steps.append(f"Mantive só as imagens com {'a tag' if len(tags) == 1 else 'as tags'} {lista}{sobra}.")
    tempo = f", em {_tempo(ms)}" if ms is not None else ""
    if results:
        nar.steps.append(f"Ordenei pelo score e separei as {len(results)} mais próximas{tempo}.")

    if not results:
        nar.tone = "warn"
        nar.body = ["Não encontrei nenhuma imagem com todas essas tags juntas. Que tal remover uma delas ou "
                    "deixar a busca mais ampla?"]
        return nar

    s1, c1, _ = results[0]
    s2 = results[1][0] if len(results) > 1 else None
    nar.body.append(rng.choice(openers or ["Pronto!", "Aqui está.", "Encontrei estes resultados.", "Feito!"]) +
                    f" O mais próximo foi \"{_clean(c1)}\", com score {fmt(s1, 3)}.")
    if (m := _margin_sentence(s1, s2)):
        nar.body.append(m)
    nar.body += _terms_sentences(query, [c for _, c, _ in results])
    if (t := _tag_sentence([ts for _, _, ts in results], tags)):
        nar.body.append(t)
    nar.body.append("Lembrando que o score serve para ordenar, não é uma porcentagem de certeza. " + rng.choice([
        "Quer tentar uma versão mais específica da busca?",
        "Se não for bem isso, descreva cor, lugar ou ação para eu afinar.",
        "Posso ajudar a refinar a busca, se quiser.",
    ]))
    return nar


# ---------------------------------------------------------------------------
# Imagem -> Imagem
# ---------------------------------------------------------------------------
def explain_image_search(
    results: list[tuple[float, str, list[str]]],
    *,
    model_name: str = "",
    dim: int | None = None,
    n_catalog: int | None = None,
    tags: list[str] | None = None,
    n_after_filter: int | None = None,
    ms: float | None = None,
    openers: list[str] | None = None,
) -> Narration:
    rng = _rng("i2i", tuple(round(r[0], 4) for r in results[:3]))
    nar = Narration()
    vec = f"um vetor de {dim} dimensões" if dim else "um vetor"
    nar.steps.append(f"Transformei a sua foto em {vec}" + (f" com o modelo {model_name}." if model_name else "."))
    if n_catalog:
        nar.steps.append(f"Comparei com as {fmt(n_catalog, 0)} imagens do catálogo usando a similaridade de cosseno.")
    if tags:
        lista = join_pt([f"\"{t}\"" for t in tags])
        sobra = f": sobraram {fmt(n_after_filter, 0)} imagens" if n_after_filter is not None else ""
        nar.steps.append(f"Mantive só as imagens com {'a tag' if len(tags) == 1 else 'as tags'} {lista}{sobra}.")
    tempo = f", em {_tempo(ms)}" if ms is not None else ""
    if results:
        nar.steps.append(f"Ordenei pela similaridade e separei as {len(results)} mais parecidas{tempo}.")

    if not results:
        nar.tone = "warn"
        nar.body = ["Não encontrei imagens com essas tags. Tente remover o filtro."]
        return nar

    s1, c1, _ = results[0]
    body = []
    if s1 > 0.99:
        body.append("A sua foto parece ser uma imagem do próprio catálogo: o 1º resultado é praticamente idêntico "
                    f"a ela (score {fmt(s1, 3)}), descrito como \"{_clean(c1)}\".")
        rest = results[1:]
        if rest:
            body.append(f"As outras imagens são as mais parecidas, começando por \"{_clean(rest[0][1])}\" "
                        f"({fmt(rest[0][0], 3)}).")
    else:
        body.append(rng.choice(openers or ["Pronto!", "Aqui está.", "Feito!"]) +
                    f" A imagem mais parecida é \"{_clean(c1)}\", com similaridade {fmt(s1, 3)}.")
        if s1 < 0.5:
            body.append("A similaridade está baixa, então o catálogo talvez não tenha nada muito parecido com a sua "
                        "foto; os resultados compartilham só alguns elementos, como cores ou cenário.")
        rest = results[1:]
    words = Counter(w for _, c, _ in rest or results for w in content_words(c))
    common = [w for w, n in words.most_common(3) if n >= 2]
    if common:
        body.append("Nas legendas dos resultados se repetem " + ", ".join(f"\"{w}\"" for w in common) +
                    ", o que dá uma boa pista do que o modelo enxergou na foto.")
    if (t := _tag_sentence([ts for _, _, ts in results], tags)):
        body.append(t)
    body.append("Entre imagens, a similaridade costuma ser bem mais alta do que entre texto e imagem, então não "
                "compare esses scores com os da busca por texto.")
    nar.body = body
    return nar


# ---------------------------------------------------------------------------
# Imagem -> Texto
# ---------------------------------------------------------------------------
def explain_image_to_text(
    captions: list[tuple[float, str]],
    zero_shot: dict[str, float] | None,
    *,
    is_siglip: bool = True,
    model_name: str = "",
    ms: float | None = None,
) -> Narration:
    nar = Narration()
    nar.steps.append("Transformei a sua foto em um vetor" + (f" com o modelo {model_name}." if model_name else "."))
    nar.steps.append("Comparei com todas as legendas do catálogo e separei as mais próximas, sem repetições.")
    if zero_shot:
        nar.steps.append(f"Comparei a foto com a frase \"uma foto de ...\" para cada um dos {len(zero_shot)} rótulos.")
    if ms is not None:
        nar.steps[-1] = nar.steps[-1].rstrip(".") + f", em {_tempo(ms)} no total."

    if captions:
        s1, c1 = captions[0]
        nar.body.append(f"A legenda que mais combina com a foto é \"{_clean(c1)}\" (score {fmt(s1, 3)}).")
        words = Counter(w for _, c in captions for w in content_words(c))
        common = [w for w, n in words.most_common(3) if n >= 2]
        if common:
            nar.body.append("As legendas seguintes concordam em " + join_pt([f"\"{w}\"" for w in common]) +
                            ", o que reforça essa leitura da imagem.")
    if zero_shot:
        ranked = sorted(zero_shot.items(), key=lambda kv: -kv[1])
        top, p = ranked[0]
        if p >= 0.6:
            nar.body.append(f"Na classificação zero-shot, o rótulo que mais combina é \"{top}\" ({_pct(p)}).")
        elif len(ranked) > 1 and ranked[1][1] >= 0.2:
            nar.body.append(f"Na classificação zero-shot, a foto fica dividida entre \"{top}\" ({_pct(p)}) e "
                            f"\"{ranked[1][0]}\" ({_pct(ranked[1][1])}); provavelmente os dois aparecem nela.")
        else:
            nar.body.append(f"Na classificação zero-shot, o rótulo mais provável é \"{top}\" ({_pct(p)}), mas sem "
                            "muita confiança: talvez nenhum rótulo descreva bem a foto.")
        nar.body.append("As porcentagens são relativas aos rótulos que você informou e somam 100%. Se nenhum "
                        "rótulo combinar com a foto, o modelo ainda assim escolhe o menos distante, então vale "
                        "incluir opções variadas.")
    elif zero_shot is not None:
        nar.body.append("Para a classificação zero-shot, informe pelo menos dois rótulos separados por vírgula.")
    return nar


# ---------------------------------------------------------------------------
# Base x Ajustado
# ---------------------------------------------------------------------------
def explain_compare(
    query: str,
    base: list[tuple[float, str, str]],
    tuned: list[tuple[float, str, str]],
) -> Narration:
    """base/tuned: [(score, legenda, id_da_imagem)]"""
    nar = Narration()
    k = len(base)
    nar.steps = [f"Rodei a busca \"{query}\" nos dois modelos, cada um com o próprio índice.",
                 f"Separei as {k} imagens mais próximas de cada um e comparei as listas."]
    ids_b, ids_t = [b[2] for b in base], [t[2] for t in tuned]
    inter = len(set(ids_b) & set(ids_t))
    if ids_b and ids_t and ids_b[0] == ids_t[0]:
        nar.body.append(f"Os dois modelos concordam no 1º lugar: \"{_clean(base[0][1])}\".")
    elif ids_b and ids_t:
        nar.body.append(f"Os modelos discordam no 1º lugar. O base escolheu \"{_clean(base[0][1])}\", e o ajustado, "
                        f"\"{_clean(tuned[0][1])}\".")
    aparecem = "aparece" if inter == 1 else "aparecem"
    nar.body.append(f"Das {k} imagens de cada lado, {inter} {aparecem} nas duas listas." +
                    (" Os resultados são bem parecidos." if inter >= max(1, k - 1) else
                     " O ajuste mudou bastante o ranking para esta busca." if inter <= k // 2 else ""))
    nar.body.append("Os scores dos dois modelos não são comparáveis entre si, porque cada um gera vetores de texto "
                    "diferentes. Compare a ordem e o conteúdo das imagens, não os números.")
    return nar
