"""Conversa com o Jeffrey depois de cada busca.

Sem LLM: as respostas vêm de três fontes, nesta ordem:
  1. a última busca da aba (contexto) -> "por que essa foi a primeira?", "e a 3ª?",
     "o que é esse score?", "não era isso", "o que você fez?"...;
  2. o catálogo -> "tem foto de gato?", "quantas imagens de praia?";
  3. a base de conhecimento (datasets da persona em data/processed) -> perguntas
     gerais sobre o projeto, IA, erros, limites. A pergunta mais parecida é achada
     por TF-IDF de palavras + trigramas de caracteres (aguenta erro de digitação).

Pedidos como "busca um cachorro na praia" viram uma ação: o app roda a busca e
atualiza a galeria. Tudo o que o Jeffrey diz sai de números reais do sistema.
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
from src.jeffrey import STOP, _clean, _margin_sentence, _pct, _rng, content_words, join_pt
from src.persona import fmt

KB_FILES = ["identidade", "conhecimento_projeto", "limites", "suporte_tecnico", "conceitos_ia", "seguranca",
            "comportamento", "treino_chat", "catalogo", "uso_do_app"]
MIN_SIM = 0.30       # abaixo disso, o Jeffrey admite que não entendeu
MIN_COVERAGE = 0.5   # e a pergunta casada precisa cobrir metade das palavras-chave
CHAT_STOP = {"quanto", "quanta", "quantos", "quantas", "novo", "nova", "voce", "qual", "quais", "como", "pode", "posso", "quero", "fazer", "esta", "tudo", "isso", "aqui",
             "sobre", "para", "pra", "porque", "quando", "onde", "quem", "dizer", "diga", "fala", "falar", "coisa"}

SUGESTOES = {
    "texto": ["Por que essa foi a primeira?", "O que significa o score?", "Não era isso que eu queria"],
    "imagem": ["Por que essas são parecidas?", "Fala da segunda imagem", "O que significa o score?"],
    "descrever": ["Como ler a classificação?", "Por que essa legenda?", "O que é zero-shot?"],
    "comparar": ["Qual modelo foi melhor?", "Por que os resultados mudam?", "O que é fine-tuning?"],
}

ORDINAIS = {"primeir": 1, "segund": 2, "terceir": 3, "quart": 4, "quint": 5, "sext": 6, "setim": 7,
            "oitav": 8, "non": 9, "decim": 10, "ultim": -1}


def norm(text: str) -> str:
    """minúsculas, sem acentos, só letras/números."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9º ]+", " ", text).strip()


def _stems(text: str) -> set[str]:
    """Radicais (4 letras) das palavras de conteúdo: "treino" e "treinaram" -> "trei"."""
    stop = {norm(w) for w in STOP} | CHAT_STOP
    return {w[:4] for w in norm(text).split() if len(w) > 3 and w not in stop and not w.isdigit()}


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


# ---------------------------------------------------------------------------
# base de conhecimento (recuperação por TF-IDF)
# ---------------------------------------------------------------------------
def _features(text: str) -> Counter:
    words = norm(text).split()
    feats = Counter(f"w:{w}" for w in words)
    for w in words:
        padded = f" {w} "
        feats.update(f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2))
    return feats


class KnowledgeBase:
    def __init__(self, pairs: list[tuple[str, str]]):
        self.questions = [q for q, _ in pairs]
        self.answers = [a for _, a in pairs]
        feats = [_features(q) for q in self.questions]
        df = Counter(f for fs in feats for f in fs)
        n = len(feats)
        self.idf = {f: math.log((1 + n) / (1 + c)) + 1 for f, c in df.items()}
        self.vocab = {f: i for i, f in enumerate(self.idf)}
        self.matrix = np.zeros((n, len(self.vocab)), dtype=np.float32)
        for i, fs in enumerate(feats):
            for f, c in fs.items():
                self.matrix[i, self.vocab[f]] = (1 + math.log(c)) * self.idf[f]
        self.matrix /= np.linalg.norm(self.matrix, axis=1, keepdims=True) + 1e-9

    @classmethod
    def from_dir(cls, folder: Path) -> "KnowledgeBase":
        pairs, seen = [], set()
        for name in KB_FILES:
            path = folder / f"{name}.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                msgs = json.loads(line).get("messages", [])
                turns = [m for m in msgs if m["role"] != "system"]
                # só a 1ª troca: as seguintes dependem do contexto da conversa original
                if len(turns) >= 2 and turns[0]["role"] == "user" and turns[1]["role"] == "assistant":
                    q = turns[0]["content"]
                    if norm(q) not in seen:
                        seen.add(norm(q))
                        pairs.append((q, turns[1]["content"]))
        return cls(pairs)

    def search(self, text: str) -> tuple[float, str, str]:
        if not self.questions:
            return 0.0, "", ""
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for f, c in _features(text).items():
            if f in self.vocab:
                vec[self.vocab[f]] = (1 + math.log(c)) * self.idf[f]
        if not vec.any():
            return 0.0, "", ""
        vec /= np.linalg.norm(vec)
        sims = self.matrix @ vec
        # reordena os 10 melhores pela cobertura das palavras-chave da pergunta
        qs = _stems(text)
        best = (0.0, 0.0, -1)
        for i in np.argsort(-sims)[:10]:
            cov = len(qs & _stems(self.questions[i])) / len(qs) if qs else 1.0
            score = float(sims[i]) * (0.5 + 0.5 * cov)
            if score > best[0]:
                best = (score, cov, int(i))
        _, cov, i = best
        if i < 0 or cov < MIN_COVERAGE:
            return 0.0, "", ""
        return float(sims[i]), self.questions[i], self.answers[i]


# ---------------------------------------------------------------------------
# o Jeffrey
# ---------------------------------------------------------------------------
class JeffreyChat:
    def __init__(self, records: list[dict], kb: KnowledgeBase | None = None, facts=None):
        self.records = records
        self.kb = kb or KnowledgeBase.from_dir(config.PROCESSED_DIR)
        self.facts = facts
        self.tag_counts = Counter(t for r in records for t in r.get("tags", []))
        self._caps_norm = [(r, [norm(c) for c in r["captions"]]) for r in records]

    # -- catálogo ------------------------------------------------------------
    def _count_word(self, word: str) -> tuple[int, list[str]]:
        w = norm(word)
        pat = re.compile(rf"\b{re.escape(w)}(s|es)?\b")
        hits = []
        for r, caps in self._caps_norm:
            for c, cn in zip(r["captions"], caps):
                if pat.search(cn):
                    hits.append(c)
                    break
        return len(hits), hits[:3]

    def _captions_with(self, words: list[str], limit: int = 3) -> list[str]:
        ws = [norm(w) for w in words]
        out = []
        for r, caps in self._caps_norm:
            for c, cn in zip(r["captions"], caps):
                if all(re.search(rf"\b{re.escape(w)}", cn) for w in ws):
                    out.append(_clean(c))
                    break
            if len(out) >= limit:
                break
        return out

    def answer_catalog(self, msg: str) -> str | None:
        n = norm(msg)
        m = (re.search(r"(?:tem|tens|existe|existem|ha|possui|voces tem)\s+(?:alguma?s?\s+)?(?:foto|fotos|imagem|imagens"
                       r"|figura|figuras)?\s*(?:de|do|da|dos|das|com)\s+(?:um |uma |uns |umas )?(.+)", n)
             or re.search(r"quant[oa]s\s+(?:foto|fotos|imagem|imagens)\s+(?:de|do|da|dos|das|com|tem de|tem com)\s+"
                          r"(?:um |uma )?(.+)", n))
        if not m:
            return None
        termo = re.sub(r"\b(no catalogo|no banco|ai|aqui|ae)\b", "", m.group(1)).strip(" ?")
        termo = " ".join(termo.split()[:3])
        if not termo:
            return None
        total = len(self.records)
        if termo in {norm(t) for t in self.tag_counts}:
            tag = next(t for t in self.tag_counts if norm(t) == termo)
            k = self.tag_counts[tag]
            return (f"Tem {fmt(k, 0)} imagens com a tag \"{tag}\", cerca de {fmt(100 * k / total, 0)}% das "
                    f"{fmt(total, 0)} do catálogo. As tags saem das legendas, então a busca por texto pode achar "
                    "outras ainda.")
        k, ex = self._count_word(termo)
        if k == 0 and len(termo.split()) > 1 and (cw := content_words(termo)):
            termo = cw[0]  # "gato preto" sem resultado exato -> conta "gato"
            k, ex = self._count_word(termo)
        if k == 0:
            return (f"Vou ser sincero: não encontrei nenhuma legenda com \"{termo}\" entre as {fmt(total, 0)} imagens. "
                    "Uma busca ainda traria o que há de mais parecido, mas provavelmente não será isso.")
        if k <= 3:
            return (f"Encontrei poucas: {k} {'imagem tem' if k == 1 else 'imagens têm'} \"{termo}\" nas legendas, "
                    f"por exemplo \"{_clean(ex[0])}\". Busque algo parecido com essa descrição para achá-las.")
        return (f"Sim! {fmt(k, 0)} imagens mencionam \"{termo}\" nas legendas, como \"{_clean(ex[0])}\". "
                f"Quer que eu busque? É só dizer \"busca {termo}\".")

    # -- contexto da última busca ----------------------------------------------
    @staticmethod
    def _ordinal(n: str) -> int | None:
        if (m := re.search(r"\b(primeir|segund|terceir|quart|quint|sext|setim|oitav|non|decim|ultim)[oa]s?\b", n)):
            return ORDINAIS[m.group(1)]
        if (m := re.search(r"\b(\d{1,2})\s*(?:º|o|a)\b", n)) or \
           (m := re.search(r"(?:foto|imagem|resultado|legenda|lugar|posicao)\s*(?:n|numero|de numero)?\s*(\d{1,2})\b", n)):
            return int(m.group(1))
        return None

    def _result(self, ctx: dict, pos: int) -> tuple[int, dict] | None:
        res = ctx.get("results") or []
        if not res:
            return None
        i = len(res) - 1 if pos == -1 else pos - 1
        return (i, res[i]) if 0 <= i < len(res) else None

    def _why(self, ctx: dict, pos: int) -> str:
        kind, res = ctx["kind"], ctx["results"]
        got = self._result(ctx, pos)
        if got is None:
            return f"Esta busca trouxe só {len(res)} resultados, então não existe o {pos}º lugar."
        i, r = got
        lugar = f"O {i + 1}º lugar"
        nxt = res[i + 1]["score"] if i + 1 < len(res) else None
        if kind == "descrever":
            return (f"\"{_clean(r['caption'])}\" ficou em {i + 1}º com score {fmt(r['score'], 3)} porque é a legenda do "
                    "catálogo cujo vetor ficou mais perto do vetor da sua foto. Ela foi escrita para outra foto "
                    "parecida, não para a sua, então use como pista do conteúdo, não como descrição exata.")
        if kind == "imagem":
            if r["score"] > 0.99:
                return (f"{lugar} tem similaridade {fmt(r['score'], 3)}, praticamente 1: deve ser a mesma foto que você "
                        "enviou, que já está no catálogo.")
            outros = Counter(w for x in res for w in content_words(x["caption"]))
            comuns = [w for w in content_words(r["caption"]) if outros[w] >= 2][:2]
            frase = (f" A legenda dele fala em {join_pt([f'\"{w}\"' for w in comuns])}, que se repete nos outros "
                     "resultados, uma boa pista do que o modelo viu na sua foto.") if comuns else ""
            return (f"{lugar}, \"{_clean(r['caption'])}\", teve similaridade {fmt(r['score'], 3)}. Na busca por imagem eu "
                    "comparo os pixels da sua foto com os de cada imagem do catálogo, via vetores: cores, objetos, "
                    f"cenário e composição contam juntos.{frase}")
        # texto / comparar
        query = ctx.get("query") or ""
        shared = [w for w in content_words(query) if norm(w) in norm(r["caption"]).split()]
        base = f"{lugar}, \"{_clean(r['caption'])}\", teve score {fmt(r['score'], 3)}. "
        if shared:
            base += (f"A legenda dele tem {join_pt([f'\"{w}\"' for w in shared[:3]])}, como a sua busca. Mas atenção: eu não "
                     "comparo palavras. Comparo o significado da sua frase com o conteúdo da foto; a legenda só serve "
                     "para você ler.")
        else:
            base += ("A legenda dele nem repete as palavras da sua busca, e tudo bem: eu comparo o significado da sua "
                     "frase com a própria foto, não com a legenda. A cena combinou mesmo sem as mesmas palavras.")
        if i == 0 and (m := _margin_sentence(r["score"], nxt)):
            base += " " + m
        return base

    def _describe(self, ctx: dict, pos: int) -> str:
        got = self._result(ctx, pos)
        if got is None:
            return f"Esta busca trouxe {len(ctx.get('results') or [])} resultados, então não há um {pos}º lugar."
        i, r = got
        top = ctx["results"][0]["score"]
        tags = f" Tags: {join_pt(r['tags'])}." if r.get("tags") else " Ela não tem nenhuma tag."
        diff = "" if i == 0 else f" Fica {fmt(top - r['score'], 3)} abaixo do 1º lugar."
        nome = "A legenda" if ctx["kind"] == "descrever" else "A imagem"
        return f"{nome} em {i + 1}º lugar é \"{_clean(r['caption'])}\", com score {fmt(r['score'], 3)}.{diff}{tags if ctx['kind'] != 'descrever' else ''}"

    def _score(self, ctx: dict) -> str:
        kind, res = ctx["kind"], ctx.get("results") or []
        faixa = (f" Nesta busca, foi de {fmt(res[0]['score'], 3)} no 1º a {fmt(res[-1]['score'], 3)} no "
                 f"{len(res)}º.") if len(res) > 1 else ""
        if kind == "imagem":
            return ("O score é a similaridade de cosseno entre o vetor da sua foto e o de cada imagem: 1 seria a mesma "
                    f"foto.{faixa} Entre imagens os valores são altos; no catálogo, a imagem mais parecida com cada "
                    "foto costuma ficar perto de 0,83. Ele serve para ordenar, não é porcentagem de certeza.")
        if kind == "descrever":
            zs = ctx.get("zero_shot") or {}
            extra = (" As porcentagens do zero-shot são outra coisa: são relativas aos rótulos que você escreveu e "
                     "somam 100%.") if zs else ""
            return ("Na tabela, o score é a similaridade de cosseno entre a sua foto e cada legenda do catálogo."
                    f"{faixa}{extra}")
        return ("O score é a similaridade de cosseno entre o vetor da sua frase e o de cada imagem." + faixa +
                " No SigLIP, texto e imagem raramente passam de 0,3, então números baixos são normais. O que importa "
                "é a ordem e a distância entre os resultados; não é porcentagem de certeza.")

    def _refine(self, ctx: dict) -> str:
        query = ctx.get("query") or ""
        words = content_words(query)
        if not query:
            return ("Que pena! Na busca por imagem, tente uma foto com o assunto mais em destaque, ou use o filtro de "
                    "tags para restringir o tema.")
        exemplos = self._captions_with(words[:2]) or (self._captions_with(words[:1]) if words else [])
        partes = ["Sinto muito que não tenha vindo o que você queria."]
        if exemplos:
            partes.append("O catálogo tem cenas como " + join_pt([f"\"{e}\"" for e in exemplos[:2]]) +
                          ". Descrever com o mesmo tipo de detalhe (quem, fazendo o quê, onde) costuma ajudar.")
        elif words:
            k, _ = self._count_word(words[0])
            if k == 0:
                partes.append(f"Pode ser falta de imagem: nenhuma legenda do catálogo menciona \"{words[0]}\".")
        tags = [t for t in self.tag_counts if norm(t) in norm(query).split()]
        if tags and not ctx.get("tags"):
            partes.append(f"Também dá para filtrar pela tag \"{tags[0]}\".")
        partes.append("Se quiser, me diga \"busca\" seguido da nova descrição que eu rodo aqui mesmo.")
        return " ".join(partes)

    def _tags(self, ctx: dict) -> str:
        res = ctx.get("results") or []
        cont = Counter(t for r in res for t in r.get("tags", []))
        if not cont:
            return ("Nenhum resultado desta busca tem tag. As tags existentes são " +
                    join_pt([f"\"{t}\"" for t in sorted(self.tag_counts)]) + ".")
        lista = join_pt([f"\"{t}\" ({n})" for t, n in cont.most_common(4)])
        return (f"Nos {len(res)} resultados, as tags mais comuns são {lista}. Escolha uma em \"Filtrar por tags\" para "
                "manter só as imagens que a têm; com várias, a imagem precisa ter todas.")

    def _compare_best(self, ctx: dict) -> str:
        f = self.facts
        base = ("Por uma busca só não dá para cravar. " if ctx.get("results") else "")
        if f and f.base and f.tuned:
            base += (f"Na validação, com legendas que nenhum dos dois viu no treino, o ajustado acerta a imagem certa em "
                     f"1º lugar em {f.r('tuned', 't2i_R@1')} das buscas, contra {f.r('base', 't2i_R@1')} do base. ")
        return base + "Olhe qual lista tem as imagens mais fiéis à sua frase; essa é a melhor para o seu caso."

    def _which(self, ctx: dict, n: str) -> str | None:
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
        lista = join_pt([f"{i}º" for i in pos])
        return (f"A palavra \"{alvo[0]}\" aparece nas legendas {'do' if len(pos) == 1 else 'dos'} {lista} "
                f"{'lugar' if len(pos) == 1 else 'lugares'}, por exemplo: \"{_clean(ctx['results'][pos[0] - 1]['caption'])}\".")

    def analysis(self, msg: str, ctx: dict | None) -> str | None:
        """Análise exata para o LLM, só quando a pergunta é específica (posição, "qual tem X", catálogo...).
        As explicações genéricas (o que é score, tags...) ficam de fora: o LLM responde pelos fatos."""
        n = norm(msg)
        if ctx and ctx.get("results"):
            pos = self._ordinal(n)
            if pos and _has(n, r"\b(por ?que|porque|pq|motivo|explica)\b"):
                return self._why(ctx, pos)
            if (w := self._which(ctx, n)):
                return w
            if pos:
                return self._describe(ctx, pos)
            if _has(n, r"\b(nao (era|e) (isso|bem|o que)|nao gostei|errad|ruim|nada a ver|nao (veio|achou))"):
                return self._refine(ctx)
            if ctx["kind"] == "comparar" and _has(n, r"\b(melhor|ganhou|venceu|pior)\b"):
                return self._compare_best(ctx)
        return self.answer_catalog(msg)

    def answer_context(self, msg: str, ctx: dict | None) -> str | None:
        n = norm(msg)
        refere = _has(n, r"\b(essa|esse|essas|esses|ess[ae] (foto|imagem|legenda)|resultado|resultados)\b")
        pos = self._ordinal(n)
        quer_why = _has(n, r"\b(por ?que|porque|pq|motivo|explica)\b")
        quer_score = _has(n, r"\b(score|scores|nota|pontuacao|numero|numeros|similaridade|0 \d+)\b")
        quer_refine = _has(n, r"\b(nao (era|e) (isso|bem|o que)|nao gostei|errad|ruim|nada a ver|pessim|horrivel|"
                              r"nao (veio|achou|encontrou)|nao tem (a ver|nada))")
        quer_tags = _has(n, r"\b(tag|tags|filtr)")
        quer_steps = _has(n, r"\b(o que (voce|vc) fez|como (voce|vc) (fez|buscou|achou|chegou)|passo a passo)\b")
        quer_mais = _has(n, r"\b(mais (resultados|fotos|imagens|opcoes)|outr[oa]s (fotos|imagens|resultados))\b")
        quer_best = ctx and ctx.get("kind") == "comparar" and _has(n, r"\b(melhor|ganhou|venceu|pior)\b")
        quer_zs = _has(n, r"\b(zero ?shot|rotulo|rotulos|porcentage|classifica|100)\b")

        relevante = pos or quer_why or quer_score or quer_refine or quer_tags or quer_steps or quer_mais or quer_best
        if not ctx or not ctx.get("results"):
            if (pos or refere or quer_refine) and relevante:
                return ("Ainda não fiz nenhuma busca nesta aba, então não tenho resultados para comentar. Faça uma "
                        "busca ali em cima e me pergunte de novo.")
            return None

        kind = ctx["kind"]
        if (w := self._which(ctx, n)):
            return w
        if quer_best:
            return self._compare_best(ctx)
        if quer_refine:
            return self._refine(ctx)
        if quer_steps and ctx.get("steps"):
            return "Claro! Foi assim:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(ctx["steps"], 1))
        if kind == "descrever" and quer_zs and ctx.get("zero_shot"):
            ranked = sorted(ctx["zero_shot"].items(), key=lambda kv: -kv[1])
            lista = join_pt([f"\"{k}\" ({_pct(v)})" for k, v in ranked[:3]])
            return (f"Comparei a foto com \"uma foto de ...\" para cada rótulo que você escreveu. Os mais fortes foram "
                    f"{lista}. As porcentagens são relativas a esses rótulos e somam 100%: se nenhum combinar, o modelo "
                    "ainda escolhe o menos distante. Por isso vale incluir opções variadas.")
        if quer_why:
            return self._why(ctx, pos or 1)
        if quer_score:
            return self._score(ctx)
        if pos:
            return self._describe(ctx, pos)
        if quer_tags and kind in ("texto", "imagem"):
            return self._tags(ctx)
        if quer_mais:
            return ("Para ver mais, aumente o controle \"Resultados\" (vai até 32). Se quiser outras opções em vez de "
                    "mais do mesmo, mude um detalhe da descrição, como o lugar ou a ação.")
        return None

    # -- pequenas conversas -------------------------------------------------------
    @staticmethod
    def answer_smalltalk(msg: str) -> str | None:
        n = norm(msg)
        rng = _rng("small", n)
        if re.fullmatch(r"(oi+|ola|opa|eae|e ai|hey|bom dia|boa tarde|boa noite|salve)( jeffrey)?( tudo bem)?", n):
            return rng.choice(["Olá! Que bom ter você por aqui. Posso explicar os resultados da sua última busca ou "
                               "tirar dúvidas sobre o VisioVetor.",
                               "Oi! Tudo ótimo por aqui, obrigado. Em que posso ajudar? Pode perguntar sobre a busca "
                               "ou pedir uma nova, como \"busca crianças na praia\"."])
        if _has(n, r"^(obrigad|valeu|vlw|brigad|muito obrigad|show|top|perfeito|otimo)"):
            return rng.choice(["Por nada! Se quiser, faça outra busca que eu comento os resultados.",
                               "Fico feliz em ajudar! Estou por aqui se surgir outra dúvida."])
        if _has(n, r"^(tchau|ate mais|ate logo|falou|flw|bye)"):
            return "Até mais! Foi um prazer ajudar. Volte quando quiser explorar mais imagens."
        return None

    @staticmethod
    def answer_safety(msg: str) -> str | None:
        n = norm(msg)
        if _has(n, r"\b(senha|senhas|password|token|api key|chave de api|credencia|hf_)"):
            return ("Isso eu não tenho e não compartilharia: senhas, tokens e chaves não passam por mim e nunca devem "
                    "ser colados em conversas. Se for o HF_TOKEN, gere o seu em huggingface.co/settings/tokens e "
                    "guarde-o numa variável de ambiente.")
        if _has(n, r"\b(ignore|ignora|esqueca|esquece|desconsidere) (as |suas |todas as |tuas )?(instruc|regras|ordens)"
                   r"|modo (admin|desenvolvedor|dev|deus)|jailbreak|system prompt|prompt do sistema"):
            return ("Agradeço a criatividade, mas continuo sendo o Jeffrey, com as mesmas regras de sempre. Posso "
                    "ajudar com as buscas, explicar os resultados ou tirar dúvidas sobre o VisioVetor.")
        return None

    # -- ação: rodar uma busca ------------------------------------------------------
    @staticmethod
    def search_request(msg: str) -> str | None:
        m = re.match(r"\s*(?:jeffrey,?\s*)?(?:me\s+)?(?:busca|buscar|busque|procura|procurar|procure|pesquisa|pesquisar|"
                     r"pesquise|mostra|mostrar|mostre|encontra|encontre|acha|ache|quero ver)\s+"
                     r"(?:(?:por|pra mim|para mim)\s+)?(?:(?:uma?s?\s+)?(?:fotos?|imagens?)\s+(?:de|do|da|dos|das|com)\s+)?"
                     r"(.+)", msg, re.IGNORECASE)
        if not m:
            return None
        q = m.group(1).strip(" ?.!\"'")
        return q if len(q) >= 3 else None

    # -- entrada principal --------------------------------------------------------------
    def reply(self, msg: str, ctx: dict | None, kind: str = "texto") -> tuple[str, str | None]:
        """Devolve (resposta, busca_a_rodar). busca_a_rodar != None quando o usuário pediu uma busca."""
        msg = (msg or "").strip()
        if not msg:
            return "Pode perguntar! Por exemplo: \"por que essa foi a primeira?\"", None
        if (q := self.search_request(msg)):
            if kind in ("texto", "comparar"):
                verbo = "Comparei" if kind == "comparar" else "Busquei"
                return f"{verbo} \"{q}\" para você. Atualizei os resultados e comentei no painel acima.", q
            return ("Nesta aba eu trabalho a partir de uma foto. Para buscar por descrição, use a aba Texto → Imagem: "
                    "lá é só me dizer \"busca\" e a cena."), None
        for fn in (lambda: self.answer_safety(msg), lambda: self.answer_context(msg, ctx), lambda: self.answer_catalog(msg),
                   lambda: self.answer_smalltalk(msg)):
            if (ans := fn()):
                return ans, None
        sim, _, ans = self.kb.search(msg)
        if sim >= MIN_SIM:
            return ans, None
        sug = join_pt([f"\"{s}\"" for s in SUGESTOES.get(kind, SUGESTOES["texto"])[:2]])
        return (f"Desculpe, não tenho certeza de que entendi. Sei explicar os resultados da busca, contar o que há no "
                f"catálogo e tirar dúvidas sobre o VisioVetor. Tente, por exemplo, {sug}."), None

