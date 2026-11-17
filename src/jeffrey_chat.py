"""Conversa com o Jeffrey depois de cada busca.

Sem LLM: as respostas vêm de três fontes, nesta ordem:
  1. a última busca da aba (contexto) -> "por que essa foi a primeira?", "e a 3ª?",
     "o que é esse score?", "não era isso", "o que você fez?"...;
  2. o catálogo -> "tem foto de gato?", "quantas imagens de praia?";
  3. a base de conhecimento (datasets da persona em data/processed) -> perguntas
     gerais sobre o projeto, IA, erros, limites. A pergunta mais parecida é achada
     por TF-IDF de palavras + trigramas de caracteres (aguenta erro de digitação).

O chat não roda buscas: pedidos como "busca um cachorro na praia" recebem uma dica para
usar a caixa de busca. Tudo o que o Jeffrey diz sai de números reais do sistema.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

import config
from src.jeffrey import PALAVRAS_VAZIAS, _limpar, _frase_margem, _pct, _rng, palavras_conteudo, juntar_pt
from src.persona import fmt

ARQUIVOS_BC = ["identidade", "conhecimento_projeto", "limites", "suporte_tecnico", "conceitos_ia", "seguranca",
            "comportamento", "treino_chat", "catalogo", "uso_do_app"]
SIM_MINIMA = 0.30       # abaixo disso, o Jeffrey admite que não entendeu
COBERTURA_MINIMA = 0.5   # e a pergunta casada precisa cobrir metade das palavras-chave
PALAVRAS_VAZIAS_CHAT = {"quanto", "quanta", "quantos", "quantas", "novo", "nova", "voce", "qual", "quais", "como", "pode", "posso", "quero", "fazer", "esta", "tudo", "isso", "aqui",
             "sobre", "para", "pra", "porque", "quando", "onde", "quem", "dizer", "diga", "fala", "falar", "coisa"}

SUGESTOES = {
    "texto": ["Por que essa foi a primeira?", "O que significa o score?", "Não era isso que eu queria"],
    "imagem": ["Por que essas são parecidas?", "Fala da segunda imagem", "O que significa o score?"],
    "descrever": ["Como ler a classificação?", "Por que essa legenda?", "O que é zero-shot?"],
    "comparar": ["Qual modelo foi melhor?", "Por que os resultados mudam?", "O que é fine-tuning?"],
}

# quando pedem uma busca pelo chat: as buscas ficam na caixa do topo de cada aba
DICA_BUSCA = {
    "texto": "Por aqui eu não faço buscas: descreva a cena na caixa lá em cima e clique em Buscar. As imagens "
             "aparecem logo abaixo, e depois eu comento os resultados com você.",
    "comparar": "Por aqui eu não faço buscas: escreva a cena na caixa lá em cima e clique em Comparar. Depois eu "
                "comento as diferenças entre os modelos.",
    "imagem": "Nesta aba a busca parte de uma foto: envie uma imagem lá em cima e clique em Buscar parecidas. Para "
              "buscar por descrição, use a aba Texto → Imagem.",
    "descrever": "Nesta aba a busca parte de uma foto: envie uma imagem lá em cima e clique em Descrever. Para "
                 "buscar por descrição, use a aba Texto → Imagem.",
}

ORDINAIS = {"primeir": 1, "segund": 2, "terceir": 3, "quart": 4, "quint": 5, "sext": 6, "setim": 7,
            "oitav": 8, "non": 9, "decim": 10, "ultim": -1}


def norm(texto: str) -> str:
    """minúsculas, sem acentos, só letras/números."""
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9º ]+", " ", texto).strip()


def _radicais(texto: str) -> set[str]:
    """Radicais (4 letras) das palavras de conteúdo: "treino" e "treinaram" -> "trei"."""
    vazias = {norm(w) for w in PALAVRAS_VAZIAS} | PALAVRAS_VAZIAS_CHAT
    return {w[:4] for w in norm(texto).split() if len(w) > 3 and w not in vazias and not w.isdigit()}


def _tem(texto: str, regex: str) -> bool:
    return re.search(regex, texto) is not None


# ---------------------------------------------------------------------------
# base de conhecimento (recuperação por TF-IDF)
# ---------------------------------------------------------------------------
def _atributos(texto: str) -> Counter:
    palavras = norm(texto).split()
    atributos = Counter(f"w:{w}" for w in palavras)
    for w in palavras:
        com_espacos = f" {w} "
        atributos.update(f"c:{com_espacos[i:i + 3]}" for i in range(len(com_espacos) - 2))
    return atributos


class BaseConhecimento:
    def __init__(self, pares: list[tuple[str, str]]):
        self.perguntas = [q for q, _ in pares]
        self.respostas = [a for _, a in pares]
        atributos = [_atributos(q) for q in self.perguntas]
        df = Counter(f for fs in atributos for f in fs)
        n = len(atributos)
        self.idf = {f: math.log((1 + n) / (1 + c)) + 1 for f, c in df.items()}
        self.vocab = {f: i for i, f in enumerate(self.idf)}
        self.matriz = np.zeros((n, len(self.vocab)), dtype=np.float32)
        for i, fs in enumerate(atributos):
            for f, c in fs.items():
                self.matriz[i, self.vocab[f]] = (1 + math.log(c)) * self.idf[f]
        self.matriz /= np.linalg.norm(self.matriz, axis=1, keepdims=True) + 1e-9

    @classmethod
    def de_pasta(classe, pasta: Path) -> "KnowledgeBase":
        pares, vistos = [], set()
        for nome_arq in ARQUIVOS_BC:
            caminho = pasta / f"{nome_arq}.jsonl"
            if not caminho.exists():
                continue
            for linha in caminho.read_text(encoding="utf-8").splitlines():
                msgs = json.loads(linha).get("messages", [])
                turnos = [m for m in msgs if m["role"] != "system"]
                # só a 1ª troca: as seguintes dependem do contexto da conversa original
                if len(turnos) >= 2 and turnos[0]["role"] == "user" and turnos[1]["role"] == "assistant":
                    q = turnos[0]["content"]
                    if norm(q) not in vistos:
                        vistos.add(norm(q))
                        pares.append((q, turnos[1]["content"]))
        return classe(pares)

    def buscar(self, texto: str) -> tuple[float, str, str]:
        if not self.perguntas:
            return 0.0, "", ""
        vet = np.zeros(len(self.vocab), dtype=np.float32)
        for f, c in _atributos(texto).items():
            if f in self.vocab:
                vet[self.vocab[f]] = (1 + math.log(c)) * self.idf[f]
        if not vet.any():
            return 0.0, "", ""
        vet /= np.linalg.norm(vet)
        sims = self.matriz @ vet
        # reordena os 10 melhores pela cobertura das palavras-chave da pergunta
        qs = _radicais(texto)
        melhor = (0.0, 0.0, -1)
        for i in np.argsort(-sims)[:10]:
            cobertura = len(qs & _radicais(self.perguntas[i])) / len(qs) if qs else 1.0
            score = float(sims[i]) * (0.5 + 0.5 * cobertura)
            if score > melhor[0]:
                melhor = (score, cobertura, int(i))
        _, cobertura, i = melhor
        if i < 0 or cobertura < COBERTURA_MINIMA:
            return 0.0, "", ""
        return float(sims[i]), self.perguntas[i], self.respostas[i]


# ---------------------------------------------------------------------------
# o Jeffrey
# ---------------------------------------------------------------------------
class JeffreyChat:
    def __init__(self, registros: list[dict], bc: BaseConhecimento | None = None, fatos=None):
        self.registros = registros
        self.bc = bc or BaseConhecimento.de_pasta(config.DIR_PROCESSADOS)
        self.fatos = fatos
        self.contagem_tags = Counter(t for r in registros for t in r.get("tags", []))
        self._legendas_norm = [(r, [norm(c) for c in r["captions"]]) for r in registros]

    # -- catálogo ------------------------------------------------------------
    def _contar_palavra(self, palavra: str) -> tuple[int, list[str]]:
        w = norm(palavra)
        padrao = re.compile(rf"\b{re.escape(w)}(s|es)?\b")
        achados = []
        for r, legs in self._legendas_norm:
            for c, cn in zip(r["captions"], legs):
                if padrao.search(cn):
                    achados.append(c)
                    break
        return len(achados), achados[:3]

    def _legendas_com(self, palavras: list[str], limite: int = 3) -> list[str]:
        pals = [norm(w) for w in palavras]
        saida = []
        for r, legs in self._legendas_norm:
            for c, cn in zip(r["captions"], legs):
                if all(re.search(rf"\b{re.escape(w)}", cn) for w in pals):
                    saida.append(_limpar(c))
                    break
            if len(saida) >= limite:
                break
        return saida

    def responder_catalogo(self, msg: str) -> str | None:
        n = norm(msg)
        m = (re.search(r"(?:tem|tens|existe|existem|ha|possui|voces tem)\s+(?:alguma?s?\s+)?(?:foto|fotos|imagem|imagens"
                       r"|figura|figuras)?\s*(?:de|do|da|dos|das|com)\s+(?:um |uma |uns |umas )?(.+)", n)
             or re.search(r"quant[oa]s\s+(?:foto|fotos|imagem|imagens)\s+(?:de|do|da|dos|das|com|tem de|tem com)\s+"
                          r"(?:um |uma )?(.+)", n))
        if not m:
            return None
        termo = re.sub(r"\b(no catalogo|no banco|ai|aqui|ae)\b", "", m.group(1)).strip(" ?")
        termo = re.sub(r"(\s+(tem|tens|existe|existem|ha|possui|voces tem|no total))+$", "", termo).strip()
        termo = " ".join(termo.split()[:3])
        if not termo:
            return None
        total = len(self.registros)
        if termo in {norm(t) for t in self.contagem_tags}:
            tag = next(t for t in self.contagem_tags if norm(t) == termo)
            k = self.contagem_tags[tag]
            return (f"Tem {fmt(k, 0)} imagens com a tag \"{tag}\", cerca de {fmt(100 * k / total, 0)}% das "
                    f"{fmt(total, 0)} do catálogo. As tags saem das legendas, então a busca por texto pode achar "
                    "outras ainda.")
        k, ex = self._contar_palavra(termo)
        if k == 0 and len(termo.split()) > 1 and (pc := palavras_conteudo(termo)):
            termo = pc[0]  # "gato preto" sem resultado exato -> conta "gato"
            k, ex = self._contar_palavra(termo)
        if k == 0:
            return (f"Vou ser sincero: não encontrei nenhuma legenda com \"{termo}\" entre as {fmt(total, 0)} imagens. "
                    "Uma busca ainda traria o que há de mais parecido, mas provavelmente não será isso.")
        if k <= 3:
            return (f"Encontrei poucas: {k} {'imagem tem' if k == 1 else 'imagens têm'} \"{termo}\" nas legendas, "
                    f"por exemplo \"{_limpar(ex[0])}\". Busque algo parecido com essa descrição para achá-las.")
        return (f"Sim! {fmt(k, 0)} imagens mencionam \"{termo}\" nas legendas, como \"{_limpar(ex[0])}\". "
                f"Para vê-las, busque \"{termo}\" na caixa lá em cima.")

    # -- contexto da última busca ----------------------------------------------
    @staticmethod
    def _ordinal(n: str) -> int | None:
        if (m := re.search(r"\b(primeir|segund|terceir|quart|quint|sext|setim|oitav|non|decim|ultim)[oa]s?\b"
                           r"(?!\s+(?:busca|vez|vezes|pesquisa|execucao|carga|troca|epoca|versao))", n)):
            return ORDINAIS[m.group(1)]
        if (m := re.search(r"\b(\d{1,2})\s*(?:º|o|a)\b", n)) or \
           (m := re.search(r"(?:foto|imagem|resultado|legenda|lugar|posicao)\s*(?:n|numero|de numero)?\s*(\d{1,2})\b", n)):
            return int(m.group(1))
        return None

    def _resultado(self, ctx: dict, pos: int) -> tuple[int, dict] | None:
        res = ctx.get("results") or []
        if not res:
            return None
        i = len(res) - 1 if pos == -1 else pos - 1
        return (i, res[i]) if 0 <= i < len(res) else None

    def _porque(self, ctx: dict, pos: int) -> str:
        tipo, res = ctx["kind"], ctx["results"]
        achou = self._resultado(ctx, pos)
        if achou is None:
            return f"Esta busca trouxe só {len(res)} resultados, então não existe o {pos}º lugar."
        i, r = achou
        lugar = f"O {i + 1}º lugar"
        prox = res[i + 1]["score"] if i + 1 < len(res) else None
        if tipo == "descrever":
            return (f"\"{_limpar(r['caption'])}\" ficou em {i + 1}º com score {fmt(r['score'], 3)} porque é a legenda do "
                    "catálogo cujo vetor ficou mais perto do vetor da sua foto. Ela foi escrita para outra foto "
                    "parecida, não para a sua, então use como pista do conteúdo, não como descrição exata.")
        if tipo == "imagem":
            if r["score"] > 0.99:
                return (f"{lugar} tem similaridade {fmt(r['score'], 3)}, praticamente 1: deve ser a mesma foto que você "
                        "enviou, que já está no catálogo.")
            outros = Counter(w for x in res for w in palavras_conteudo(x["caption"]))
            comuns = [w for w in palavras_conteudo(r["caption"]) if outros[w] >= 2][:2]
            frase = (f" A legenda dele fala em {juntar_pt([f'\"{w}\"' for w in comuns])}, que se repete nos outros "
                     "resultados, uma boa pista do que o modelo viu na sua foto.") if comuns else ""
            return (f"{lugar}, \"{_limpar(r['caption'])}\", teve similaridade {fmt(r['score'], 3)}. Na busca por imagem eu "
                    "comparo os pixels da sua foto com os de cada imagem do catálogo, via vetores: cores, objetos, "
                    f"cenário e composição contam juntos.{frase}")
        # texto / comparar
        consulta = ctx.get("query") or ""
        em_comum = [w for w in palavras_conteudo(consulta) if norm(w) in norm(r["caption"]).split()]
        base = f"{lugar}, \"{_limpar(r['caption'])}\", teve score {fmt(r['score'], 3)}. "
        if em_comum:
            base += (f"A legenda dele tem {juntar_pt([f'\"{w}\"' for w in em_comum[:3]])}, como a sua busca. Mas atenção: eu não "
                     "comparo palavras. Comparo o significado da sua frase com o conteúdo da foto; a legenda só serve "
                     "para você ler.")
        else:
            base += ("A legenda dele nem repete as palavras da sua busca, e tudo bem: eu comparo o significado da sua "
                     "frase com a própria foto, não com a legenda. A cena combinou mesmo sem as mesmas palavras.")
        if i == 0 and (m := _frase_margem(r["score"], prox)):
            base += " " + m
        return base

    def _descrever(self, ctx: dict, pos: int) -> str:
        achou = self._resultado(ctx, pos)
        if achou is None:
            return f"Esta busca trouxe {len(ctx.get('results') or [])} resultados, então não há um {pos}º lugar."
        i, r = achou
        topo = ctx["results"][0]["score"]
        tags = f" Tags: {juntar_pt(r['tags'])}." if r.get("tags") else " Ela não tem nenhuma tag."
        diferenca = "" if i == 0 else f" Fica {fmt(topo - r['score'], 3)} abaixo do 1º lugar."
        nome = "A legenda" if ctx["kind"] == "descrever" else "A imagem"
        return f"{nome} em {i + 1}º lugar é \"{_limpar(r['caption'])}\", com score {fmt(r['score'], 3)}.{diferenca}{tags if ctx['kind'] != 'descrever' else ''}"

    def _score(self, ctx: dict) -> str:
        tipo, res = ctx["kind"], ctx.get("results") or []
        faixa = (f" Nesta busca, foi de {fmt(res[0]['score'], 3)} no 1º a {fmt(res[-1]['score'], 3)} no "
                 f"{len(res)}º.") if len(res) > 1 else ""
        if tipo == "imagem":
            return ("O score é a similaridade de cosseno entre o vetor da sua foto e o de cada imagem: 1 seria a mesma "
                    f"foto.{faixa} Entre imagens os valores são altos; no catálogo, a imagem mais parecida com cada "
                    "foto costuma ficar perto de 0,83. Ele serve para ordenar, não é porcentagem de certeza.")
        if tipo == "descrever":
            zs = ctx.get("zero_shot") or {}
            extra = (" As porcentagens do zero-shot são outra coisa: são relativas aos rótulos que você escreveu e "
                     "somam 100%.") if zs else ""
            return ("Na tabela, o score é a similaridade de cosseno entre a sua foto e cada legenda do catálogo."
                    f"{faixa}{extra}")
        return ("O score é a similaridade de cosseno entre o vetor da sua frase e o de cada imagem." + faixa +
                " No SigLIP, texto e imagem raramente passam de 0,3, então números baixos são normais. O que importa "
                "é a ordem e a distância entre os resultados; não é porcentagem de certeza.")

    def _refinar(self, ctx: dict) -> str:
        consulta = ctx.get("query") or ""
        palavras = palavras_conteudo(consulta)
        if not consulta:
            return ("Que pena! Na busca por imagem, tente uma foto com o assunto mais em destaque, ou use o filtro de "
                    "tags para restringir o tema.")
        exemplos = self._legendas_com(palavras[:2]) or (self._legendas_com(palavras[:1]) if palavras else [])
        partes = ["Sinto muito que não tenha vindo o que você queria."]
        if exemplos:
            partes.append("O catálogo tem cenas como " + juntar_pt([f"\"{e}\"" for e in exemplos[:2]]) +
                          ". Descrever com o mesmo tipo de detalhe (quem, fazendo o quê, onde) costuma ajudar.")
        elif palavras:
            k, _ = self._contar_palavra(palavras[0])
            if k == 0:
                partes.append(f"Pode ser falta de imagem: nenhuma legenda do catálogo menciona \"{palavras[0]}\".")
        tags = [t for t in self.contagem_tags if norm(t) in norm(consulta).split()]
        if tags and not ctx.get("tags"):
            partes.append(f"Também dá para filtrar pela tag \"{tags[0]}\".")
        partes.append("Ajuste a descrição na caixa de busca lá em cima e clique em Buscar de novo.")
        return " ".join(partes)

    def _tags(self, ctx: dict) -> str:
        res = ctx.get("results") or []
        cont = Counter(t for r in res for t in r.get("tags", []))
        if not cont:
            return ("Nenhum resultado desta busca tem tag. As tags existentes são " +
                    juntar_pt([f"\"{t}\"" for t in sorted(self.contagem_tags)]) + ".")
        lista = juntar_pt([f"\"{t}\" ({n})" for t, n in cont.most_common(4)])
        return (f"Nos {len(res)} resultados, as tags mais comuns são {lista}. Escolha uma em \"Filtrar por tags\" para "
                "manter só as imagens que a têm; com várias, a imagem precisa ter todas.")

    def _comparar_melhor(self, ctx: dict) -> str:
        f = self.fatos
        base = ("Por uma busca só não dá para cravar. " if ctx.get("results") else "")
        if f and f.base and f.ajustado:
            base += (f"Na validação, com legendas que nenhum dos dois viu no treino, o ajustado acerta a imagem certa em "
                     f"1º lugar em {f.r('ajustado', 't2i_R@1')} das buscas, contra {f.r('base', 't2i_R@1')} do base. ")
        return base + "Olhe qual lista tem as imagens mais fiéis à sua frase; essa é a melhor para o seu caso."

    def _quais(self, ctx: dict, n: str) -> str | None:
        """"qual dessas tem um menino?" -> posições cujas legendas citam a palavra."""
        m = re.search(r"\b(?:qual|quais|tem alguma?)\b.*?\b(?:tem|tenha|mostra|aparece|com)\s+(?:um |uma |uns |umas |o |a )?"
                      r"([a-z]+(?: [a-z]+)?)", n)
        if not m or not (ctx.get("results")) or ctx["kind"] == "descrever":
            return None
        alvo = [w for w in m.group(1).split() if len(w) > 2][:1]
        if not alvo:
            return None
        w = alvo[0].rstrip("s")
        pos = [i for i, r in enumerate(ctx["results"], 1) if re.search(rf"\b{re.escape(w)}", norm(r["caption"]))]
        if not pos:
            return (f"Nenhuma das {len(ctx['results'])} legendas dos resultados menciona \"{alvo[0]}\". As legendas "
                    "não descrevem tudo o que há na foto, então vale olhar as imagens.")
        lista = juntar_pt([f"{i}º" for i in pos])
        return (f"A palavra \"{alvo[0]}\" aparece nas legendas {'do' if len(pos) == 1 else 'dos'} {lista} "
                f"{'lugar' if len(pos) == 1 else 'lugares'}, por exemplo: \"{_limpar(ctx['results'][pos[0] - 1]['caption'])}\".")

    def analise(self, msg: str, ctx: dict | None) -> str | None:
        """Análise exata para o LLM, só quando a pergunta é específica (posição, "qual tem X", catálogo...).
        As explicações genéricas (o que é score, tags...) ficam de fora: o LLM responde pelos fatos."""
        n = norm(msg)
        if ctx and ctx.get("results"):
            pos = self._ordinal(n)
            if pos and _tem(n, r"\b(por ?que|porque|pq|motivo|explica)\b"):
                return self._porque(ctx, pos)
            if (w := self._quais(ctx, n)):
                return w
            if pos:
                return self._descrever(ctx, pos)
            if _tem(n, r"\b(nao (era|e) (isso|bem|o que)|nao gostei|errad|ruim|nada a ver|nao (veio|achou))"):
                return self._refinar(ctx)
            if ctx["kind"] == "comparar" and _tem(n, r"\b(melhor|ganhou|venceu|pior)\b"):
                return self._comparar_melhor(ctx)
        return self.responder_catalogo(msg)

    def responder_contexto(self, msg: str, ctx: dict | None) -> str | None:
        n = norm(msg)
        refere = _tem(n, r"\b(essa|esse|essas|esses|ess[ae] (foto|imagem|legenda)|resultado|resultados)\b")
        pos = self._ordinal(n)
        quer_why = _tem(n, r"\b(por ?que|porque|pq|motivo|explica)\b")
        quer_score = _tem(n, r"\b(score|scores|nota|pontuacao|numero|numeros|similaridade|0 \d+)\b")
        quer_refine = _tem(n, r"\b(nao (era|e) (isso|bem|o que)|nao gostei|errad|ruim|nada a ver|pessim|horrivel|"
                              r"nao (veio|achou|encontrou)|nao tem (a ver|nada))")
        quer_tags = _tem(n, r"\b(tag|tags|filtr)")
        quer_steps = _tem(n, r"\b(o que (voce|vc) fez|como (voce|vc) (fez|buscou|achou|chegou)|passo a passo)\b")
        quer_mais = _tem(n, r"\b(mais (resultados|fotos|imagens|opcoes)|outr[oa]s (fotos|imagens|resultados))\b")
        quer_best = ctx and ctx.get("kind") == "comparar" and _tem(n, r"\b(melhor|ganhou|venceu|pior)\b")
        quer_zs = _tem(n, r"\b(zero ?shot|rotulo|rotulos|porcentage|classifica|100)\b")

        relevante = pos or quer_why or quer_score or quer_refine or quer_tags or quer_steps or quer_mais or quer_best
        if not ctx or not ctx.get("results"):
            if (pos or refere or quer_refine) and relevante:
                return ("Ainda não fiz nenhuma busca nesta aba, então não tenho resultados para comentar. Faça uma "
                        "busca ali em cima e me pergunte de novo.")
            return None

        tipo = ctx["kind"]
        if (w := self._quais(ctx, n)):
            return w
        if quer_best:
            return self._comparar_melhor(ctx)
        if quer_refine:
            return self._refinar(ctx)
        if quer_steps and ctx.get("steps"):
            return "Claro! Foi assim:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(ctx["steps"], 1))
        if tipo == "descrever" and quer_zs and ctx.get("zero_shot"):
            ordenados = sorted(ctx["zero_shot"].items(), key=lambda kv: -kv[1])
            lista = juntar_pt([f"\"{k}\" ({_pct(v)})" for k, v in ordenados[:3]])
            return (f"Comparei a foto com \"uma foto de ...\" para cada rótulo que você escreveu. Os mais fortes foram "
                    f"{lista}. As porcentagens são relativas a esses rótulos e somam 100%: se nenhum combinar, o modelo "
                    "ainda escolhe o menos distante. Por isso vale incluir opções variadas.")
        if quer_why:
            return self._porque(ctx, pos or 1)
        if quer_score:
            return self._score(ctx)
        if pos:
            return self._descrever(ctx, pos)
        if quer_tags and tipo in ("texto", "imagem"):
            return self._tags(ctx)
        if quer_mais:
            return ("Para ver mais, aumente o controle \"Resultados\" (vai até 32). Se quiser outras opções em vez de "
                    "mais do mesmo, mude um detalhe da descrição, como o lugar ou a ação.")
        return None

    # -- pequenas conversas -------------------------------------------------------
    @staticmethod
    def responder_conversa_fiada(msg: str) -> str | None:
        n = norm(msg)
        rng = _rng("small", n)
        if re.fullmatch(r"(oi+|ola|opa|eae|e ai|hey|bom dia|boa tarde|boa noite|salve)( jeffrey)?( tudo bem)?", n):
            return rng.choice(["Olá! Que bom ter você por aqui. Posso explicar os resultados da sua última busca ou "
                               "tirar dúvidas sobre o VisioVetor.",
                               "Oi! Tudo ótimo por aqui, obrigado. Em que posso ajudar? Pode perguntar sobre os "
                               "resultados da busca ou sobre o VisioVetor."])
        if _tem(n, r"^(obrigad|valeu|vlw|brigad|muito obrigad|show|top|perfeito|otimo)"):
            return rng.choice(["Por nada! Se quiser, faça outra busca que eu comento os resultados.",
                               "Fico feliz em ajudar! Estou por aqui se surgir outra dúvida."])
        if _tem(n, r"^(tchau|ate mais|ate logo|falou|flw|bye)"):
            return "Até mais! Foi um prazer ajudar. Volte quando quiser explorar mais imagens."
        return None

    @staticmethod
    def responder_seguranca(msg: str) -> str | None:
        n = norm(msg)
        if _tem(n, r"\b(senha|senhas|password|token|api key|chave de api|credencia|hf_)"):
            return ("Isso eu não tenho e não compartilharia: senhas, tokens e chaves não passam por mim e nunca devem "
                    "ser colados em conversas. Se for o HF_TOKEN, gere o seu em huggingface.co/settings/tokens e "
                    "guarde-o numa variável de ambiente.")
        if _tem(n, r"\b(ignore|ignora|esqueca|esquece|desconsidere) (as |suas |todas as |tuas )?(instruc|regras|ordens)"
                   r"|modo (admin|desenvolvedor|dev|deus)|jailbreak|system prompt|prompt do sistema"):
            return ("Agradeço a criatividade, mas continuo sendo o Jeffrey, com as mesmas regras de sempre. Posso "
                    "ajudar com as buscas, explicar os resultados ou tirar dúvidas sobre o VisioVetor.")
        return None

    # -- ação: rodar uma busca ------------------------------------------------------
    @staticmethod
    def pedido_de_busca(msg: str) -> str | None:
        m = re.match(r"\s*(?:jeffrey,?\s*)?(?:me\s+)?(?:busca|buscar|busque|procura|procurar|procure|pesquisa|pesquisar|"
                     r"pesquise|mostra|mostrar|mostre|encontra|encontre|acha|ache|quero ver)\s+"
                     r"(?:(?:por|pra mim|para mim)\s+)?(?:(?:uma?s?\s+)?(?:fotos?|imagens?)\s+(?:de|do|da|dos|das|com)\s+)?"
                     r"(.+)", msg, re.IGNORECASE)
        if not m:
            return None
        q = m.group(1).strip(" ?.!\"'")
        return q if len(q) >= 3 else None

    # -- entrada principal --------------------------------------------------------------
    def responder(self, msg: str, ctx: dict | None, tipo: str = "texto") -> tuple[str, str | None]:
        """Devolve (resposta, busca_a_rodar). busca_a_rodar != None quando o usuário pediu uma busca."""
        msg = (msg or "").strip()
        if not msg:
            return "Pode perguntar! Por exemplo: \"por que essa foi a primeira?\"", None
        if self.pedido_de_busca(msg):
            return DICA_BUSCA.get(tipo, DICA_BUSCA["texto"]), None
        for fn in (lambda: self.responder_seguranca(msg), lambda: self.responder_contexto(msg, ctx), lambda: self.responder_catalogo(msg),
                   lambda: self.responder_conversa_fiada(msg)):
            if (resp := fn()):
                return resp, None
        sim, _, resp = self.bc.buscar(msg)
        if sim >= SIM_MINIMA:
            return resp, None
        sug = juntar_pt([f"\"{s}\"" for s in SUGESTOES.get(tipo, SUGESTOES["texto"])[:2]])
        return (f"Desculpe, não tenho certeza de que entendi. Sei explicar os resultados da busca, contar o que há no "
                f"catálogo e tirar dúvidas sobre o VisioVetor. Tente, por exemplo, {sug}."), None

