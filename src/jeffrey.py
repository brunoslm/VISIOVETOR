"""Jeffrey narra as buscas: o que foi feito e como ler os resultados.

Usado em dois lugares, com o mesmo texto:
  * app.py  -> painel do Jeffrey logo abaixo dos resultados (HTML);
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
from pathlib import Path

from src.persona import NOME, fmt

# mesmo avatar do chat
_AVATAR = (Path(__file__).with_name("assets") / "jeffrey.svg").read_text(encoding="utf-8").strip()

PALAVRAS_VAZIAS = set("""a o os as um uma uns umas de da do das dos em na no nas nos num numa e é ao à com por para pela pelo
que se sua seu suas seus está estão enquanto outro outra outros outras como mais muito sobre entre até lá
ele ela eles elas isso este esta esse essa ser são tem têm foto imagem imagens fotos""".split())


def juntar_pt(itens: list[str]) -> str:
    """["a", "b", "c"] -> 'a, b e c'"""
    itens = list(itens)
    return itens[0] if len(itens) == 1 else ", ".join(itens[:-1]) + " e " + itens[-1]


def palavras_conteudo(texto: str) -> list[str]:
    return list(dict.fromkeys(
        w for w in re.findall(r"\w+", texto.lower()) if len(w) > 3 and w not in PALAVRAS_VAZIAS and not w.isdigit()
    ))


def _rng(*chave: object) -> random.Random:
    """Aleatoriedade estável: a mesma busca gera sempre o mesmo texto."""
    h = hashlib.sha256("|".join(map(str, chave)).encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def _limpar(legenda: str) -> str:
    return legenda.strip().rstrip(".")


@dataclass
class Narracao:
    passos: list[str] = field(default_factory=list)  # "o que eu fiz"
    corpo: list[str] = field(default_factory=list)   # análise dos resultados
    tom: str = "info"                               # info | warn

    def texto(self, com_passos: bool = False) -> str:
        partes = []
        if com_passos and self.passos:
            partes.append("O que eu fiz:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(self.passos, 1)))
        partes.append(" ".join(self.corpo))
        return "\n\n".join(partes)

    def html(self) -> str:
        passos = ""
        if self.passos:
            itens = "".join(f"<li>{escape(s)}</li>" for s in self.passos)
            passos = (f'<details class="vv-jeffrey-steps"><summary>O que eu fiz</summary>'
                     f'<ol>{itens}</ol></details>')
        corpo = "".join(f"<p>{escape(p)}</p>" for p in self.corpo)
        return (f'<div class="vv-jeffrey vv-jeffrey-{self.tom}" role="status" aria-live="polite">'
                f'<div class="vv-jeffrey-avatar" aria-hidden="true">{_AVATAR}</div>'
                f'<div class="vv-jeffrey-content"><p class="vv-jeffrey-name">{NOME}'
                f'<span>assistente do VisioVetor</span></p>{corpo}{passos}</div></div>')


def boas_vindas(tipo: str = "texto") -> Narracao:
    msgs = {
        "texto": "Olá! Eu sou o Jeffrey. Descreva uma cena e clique em Buscar: eu conto aqui o que fiz e como "
                 "ler os resultados.",
        "imagem": "Envie uma foto e clique em Buscar parecidas. Depois eu explico o que encontrei.",
        "descrever": "Envie uma foto e clique em Descrever. Eu mostro as legendas mais próximas e explico a "
                     "classificação zero-shot.",
        "comparar": "Digite uma busca e clique em Comparar. Eu digo o que mudou entre o modelo base e o ajustado.",
    }
    return Narracao(corpo=[msgs[tipo]])


# ---------------------------------------------------------------------------
# análises reutilizáveis
# ---------------------------------------------------------------------------
def _tempo(ms: float) -> str:
    return f"{fmt(ms / 1000)} s" if ms >= 1000 else f"{fmt(ms, 0)} ms"


def _pct(p: float) -> str:
    return "menos de 1%" if p < 0.01 else f"{fmt(100 * p, 0)}%"


def _frase_margem(s1: float, s2: float | None) -> str:
    if s2 is None:
        return ""
    d = s1 - s2
    if d < 0.005:
        return (f"Ele está praticamente empatado com o 2º ({fmt(s2, 3)}), então a ordem entre os dois diz pouco: "
                "vale olhar ambos.")
    if d > 0.02:
        return f"Ele se destaca bem do 2º, que ficou em {fmt(s2, 3)}, o que indica uma correspondência clara."
    return f"O 2º vem logo atrás ({fmt(s2, 3)}), então os dois combinam de forma parecida com a busca."


def _frases_termos(consulta: str, legendas: list[str]) -> list[str]:
    palavras_consulta = palavras_conteudo(consulta)
    palavras_legendas = [set(palavras_conteudo(c)) for c in legendas]
    n = len(legendas)
    presentes = [w for w in palavras_consulta if sum(w in pc for pc in palavras_legendas) >= max(2, (n + 1) // 2)]
    ausentes = [w for w in palavras_consulta if not any(w in pc for pc in palavras_legendas)]
    saida = []
    if presentes:
        sel = presentes[:2]
        pals = " e ".join(f"\"{w}\"" for w in sel)
        verbo = "aparece" if len(sel) == 1 else "aparecem"
        artigo = "O termo" if len(sel) == 1 else "Os termos"
        saida.append(f"{artigo} {pals} {verbo} na maioria das legendas dos resultados, sinal de que o modelo captou "
                   "bem essa parte da descrição.")
    if ausentes:
        saida.append(f"Já \"{ausentes[0]}\" não aparece em nenhuma legenda dos resultados. As legendas não descrevem "
                   "tudo o que há na foto, mas, se esse detalhe for importante, vale destacá-lo na busca.")
    return saida


def _frase_tag(listas_tags: list[list[str]], tags_usadas: list[str] | None) -> str:
    contagens = Counter(t for ts in listas_tags for t in ts if not tags_usadas or t not in tags_usadas)
    if not contagens:
        return ""
    tag, qtd = contagens.most_common(1)[0]
    n = len(listas_tags)
    if n < 3 or qtd < max(3, (n + 1) // 2):
        return ""
    onde = f"em todos os {n} resultados" if qtd == n else f"em {qtd} dos {n} resultados"
    return f"Para restringir, a tag \"{tag}\" aparece {onde}."


# ---------------------------------------------------------------------------
# Texto -> Imagem
# ---------------------------------------------------------------------------
def explicar_busca_texto(
    consulta: str,
    resultados: list[tuple[float, str, list[str]]],
    *,
    nome_modelo: str = "",
    dim: int | None = None,
    n_catalogo: int | None = None,
    tags: list[str] | None = None,
    n_apos_filtro: int | None = None,
    ms: float | None = None,
    aberturas: list[str] | None = None,
) -> Narracao:
    """results: [(score, legenda, tags)] em ordem de ranking."""
    rng = _rng("t2i", consulta, len(resultados), tags)
    nar = Narracao()

    # o que eu fiz
    vet = f"um vetor de {dim} dimensões" if dim else "um vetor"
    nar.passos.append(f"Transformei \"{consulta}\" em {vet}" + (f" com o modelo {nome_modelo}." if nome_modelo else "."))
    if n_catalogo:
        nar.passos.append(f"Comparei esse vetor com as {fmt(n_catalogo, 0)} imagens do catálogo usando a "
                         "similaridade de cosseno.")
    if tags:
        lista = juntar_pt([f"\"{t}\"" for t in tags])
        sobra = f": sobraram {fmt(n_apos_filtro, 0)} imagens" if n_apos_filtro is not None else ""
        nar.passos.append(f"Mantive só as imagens com {'a tag' if len(tags) == 1 else 'as tags'} {lista}{sobra}.")
    tempo = f", em {_tempo(ms)}" if ms is not None else ""
    if resultados:
        nar.passos.append(f"Ordenei pelo score e separei as {len(resultados)} mais próximas{tempo}.")

    if not resultados:
        nar.tom = "warn"
        nar.corpo = ["Não encontrei nenhuma imagem com todas essas tags juntas. Que tal remover uma delas ou "
                    "deixar a busca mais ampla?"]
        return nar

    s1, c1, _ = resultados[0]
    s2 = resultados[1][0] if len(resultados) > 1 else None
    nar.corpo.append(rng.choice(aberturas or ["Pronto!", "Aqui está.", "Encontrei estes resultados.", "Feito!"]) +
                    f" O mais próximo foi \"{_limpar(c1)}\", com score {fmt(s1, 3)}.")
    if (m := _frase_margem(s1, s2)):
        nar.corpo.append(m)
    nar.corpo += _frases_termos(consulta, [c for _, c, _ in resultados])
    if (t := _frase_tag([ts for _, _, ts in resultados], tags)):
        nar.corpo.append(t)
    nar.corpo.append("Lembrando que o score serve para ordenar, não é uma porcentagem de certeza. " + rng.choice([
        "Quer tentar uma versão mais específica da busca?",
        "Se não for bem isso, descreva cor, lugar ou ação para eu afinar.",
        "Posso ajudar a refinar a busca, se quiser.",
    ]))
    return nar


# ---------------------------------------------------------------------------
# Imagem -> Imagem
# ---------------------------------------------------------------------------
def explicar_busca_imagem(
    resultados: list[tuple[float, str, list[str]]],
    *,
    nome_modelo: str = "",
    dim: int | None = None,
    n_catalogo: int | None = None,
    tags: list[str] | None = None,
    n_apos_filtro: int | None = None,
    ms: float | None = None,
    aberturas: list[str] | None = None,
) -> Narracao:
    rng = _rng("i2i", tuple(round(r[0], 4) for r in resultados[:3]))
    nar = Narracao()
    vet = f"um vetor de {dim} dimensões" if dim else "um vetor"
    nar.passos.append(f"Transformei a sua foto em {vet}" + (f" com o modelo {nome_modelo}." if nome_modelo else "."))
    if n_catalogo:
        nar.passos.append(f"Comparei com as {fmt(n_catalogo, 0)} imagens do catálogo usando a similaridade de cosseno.")
    if tags:
        lista = juntar_pt([f"\"{t}\"" for t in tags])
        sobra = f": sobraram {fmt(n_apos_filtro, 0)} imagens" if n_apos_filtro is not None else ""
        nar.passos.append(f"Mantive só as imagens com {'a tag' if len(tags) == 1 else 'as tags'} {lista}{sobra}.")
    tempo = f", em {_tempo(ms)}" if ms is not None else ""
    if resultados:
        nar.passos.append(f"Ordenei pela similaridade e separei as {len(resultados)} mais parecidas{tempo}.")

    if not resultados:
        nar.tom = "warn"
        nar.corpo = ["Não encontrei imagens com essas tags. Tente remover o filtro."]
        return nar

    s1, c1, _ = resultados[0]
    corpo = []
    if s1 > 0.99:
        corpo.append("A sua foto parece ser uma imagem do próprio catálogo: o 1º resultado é praticamente idêntico "
                    f"a ela (score {fmt(s1, 3)}), descrito como \"{_limpar(c1)}\".")
        demais = resultados[1:]
        if demais:
            corpo.append(f"As outras imagens são as mais parecidas, começando por \"{_limpar(demais[0][1])}\" "
                        f"({fmt(demais[0][0], 3)}).")
    else:
        corpo.append(rng.choice(aberturas or ["Pronto!", "Aqui está.", "Feito!"]) +
                    f" A imagem mais parecida é \"{_limpar(c1)}\", com similaridade {fmt(s1, 3)}.")
        if s1 < 0.5:
            corpo.append("A similaridade está baixa, então o catálogo talvez não tenha nada muito parecido com a sua "
                        "foto; os resultados compartilham só alguns elementos, como cores ou cenário.")
        demais = resultados[1:]
    palavras = Counter(w for _, c, _ in demais or resultados for w in palavras_conteudo(c))
    comuns = [w for w, n in palavras.most_common(3) if n >= 2]
    if comuns:
        corpo.append("Nas legendas dos resultados se repetem " + ", ".join(f"\"{w}\"" for w in comuns) +
                    ", o que dá uma boa pista do que o modelo enxergou na foto.")
    if (t := _frase_tag([ts for _, _, ts in resultados], tags)):
        corpo.append(t)
    corpo.append("Entre imagens, a similaridade costuma ser bem mais alta do que entre texto e imagem, então não "
                "compare esses scores com os da busca por texto.")
    nar.corpo = corpo
    return nar


# ---------------------------------------------------------------------------
# Imagem -> Texto
# ---------------------------------------------------------------------------
def explicar_imagem_para_texto(
    legendas: list[tuple[float, str]],
    zero_shot: dict[str, float] | None,
    *,
    e_siglip: bool = True,
    nome_modelo: str = "",
    ms: float | None = None,
) -> Narracao:
    nar = Narracao()
    nar.passos.append("Transformei a sua foto em um vetor" + (f" com o modelo {nome_modelo}." if nome_modelo else "."))
    nar.passos.append("Comparei com todas as legendas do catálogo e separei as mais próximas, sem repetições.")
    if zero_shot:
        nar.passos.append(f"Comparei a foto com a frase \"uma foto de ...\" para cada um dos {len(zero_shot)} rótulos.")
    if ms is not None:
        nar.passos[-1] = nar.passos[-1].rstrip(".") + f", em {_tempo(ms)} no total."

    if legendas:
        s1, c1 = legendas[0]
        nar.corpo.append(f"A legenda que mais combina com a foto é \"{_limpar(c1)}\" (score {fmt(s1, 3)}).")
        palavras = Counter(w for _, c in legendas for w in palavras_conteudo(c))
        comuns = [w for w, n in palavras.most_common(3) if n >= 2]
        if comuns:
            nar.corpo.append("As legendas seguintes concordam em " + juntar_pt([f"\"{w}\"" for w in comuns]) +
                            ", o que reforça essa leitura da imagem.")
    if zero_shot:
        ordenados = sorted(zero_shot.items(), key=lambda kv: -kv[1])
        topo, p = ordenados[0]
        if p >= 0.6:
            nar.corpo.append(f"Na classificação zero-shot, o rótulo que mais combina é \"{topo}\" ({_pct(p)}).")
        elif len(ordenados) > 1 and ordenados[1][1] >= 0.2:
            nar.corpo.append(f"Na classificação zero-shot, a foto fica dividida entre \"{topo}\" ({_pct(p)}) e "
                            f"\"{ordenados[1][0]}\" ({_pct(ordenados[1][1])}); provavelmente os dois aparecem nela.")
        else:
            nar.corpo.append(f"Na classificação zero-shot, o rótulo mais provável é \"{topo}\" ({_pct(p)}), mas sem "
                            "muita confiança: talvez nenhum rótulo descreva bem a foto.")
        nar.corpo.append("As porcentagens são relativas aos rótulos que você informou e somam 100%. Se nenhum "
                        "rótulo combinar com a foto, o modelo ainda assim escolhe o menos distante, então vale "
                        "incluir opções variadas.")
    elif zero_shot is not None:
        nar.corpo.append("Para a classificação zero-shot, informe pelo menos dois rótulos separados por vírgula.")
    return nar


# ---------------------------------------------------------------------------
# Base x Ajustado
# ---------------------------------------------------------------------------
def explicar_comparacao(
    consulta: str,
    base: list[tuple[float, str, str]],
    ajustado: list[tuple[float, str, str]],
) -> Narracao:
    """base/tuned: [(score, legenda, id_da_imagem)]"""
    nar = Narracao()
    k = len(base)
    nar.passos = [f"Rodei a busca \"{consulta}\" nos dois modelos, cada um com o próprio índice.",
                 f"Separei as {k} imagens mais próximas de cada um e comparei as listas."]
    ids_b, ids_a = [b[2] for b in base], [t[2] for t in ajustado]
    intersecao = len(set(ids_b) & set(ids_a))
    if ids_b and ids_a and ids_b[0] == ids_a[0]:
        nar.corpo.append(f"Os dois modelos concordam no 1º lugar: \"{_limpar(base[0][1])}\".")
    elif ids_b and ids_a:
        nar.corpo.append(f"Os modelos discordam no 1º lugar. O base escolheu \"{_limpar(base[0][1])}\", e o ajustado, "
                        f"\"{_limpar(ajustado[0][1])}\".")
    aparecem = "aparece" if intersecao == 1 else "aparecem"
    nar.corpo.append(f"Das {k} imagens de cada lado, {intersecao} {aparecem} nas duas listas." +
                    (" Os resultados são bem parecidos." if intersecao >= max(1, k - 1) else
                     " O ajuste mudou bastante o ranking para esta busca." if intersecao <= k // 2 else ""))
    nar.corpo.append("Os scores dos dois modelos não são comparáveis entre si, porque cada um gera vetores de texto "
                    "diferentes. Compare a ordem e o conteúdo das imagens, não os números.")
    return nar
