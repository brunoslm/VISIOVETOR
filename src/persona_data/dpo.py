"""Pares de preferência (DPO) gerados a partir das boas respostas do Jeffrey.

Para cada resposta "chosen" (na persona), criamos uma "rejected" que quebra a
persona de um jeito específico. Assim o modelo aprende o que NÃO fazer:
  * grosseria / descaso        * nome errado ("James")
  * números inventados          * excesso de emojis e caixa alta
  * resposta seca e inútil      * idioma errado (inglês sem o usuário pedir)
  * certeza exagerada           * concordar com pedido fora do escopo
"""

from __future__ import annotations

import random
import re

from src.persona_data.common import pref

RUDE = ["Isso é óbvio.", "Leia a documentação.", "Já expliquei isso antes.", "Pergunta básica, hein.",
        "Não tenho tempo pra isso.", "Se vira."]
SECO = ["ok", "sim", "não sei", "depende", "pesquisa aí", "tanto faz"]
INGLES = ["Sure! Just type what you want and click Search.", "I don't know, check the docs.",
          "That's how the model works. Anything else?"]
CERTEZA = ["Com certeza absoluta, o sistema nunca erra nisso.", "Garanto 100%, pode confiar cegamente.",
           "Não existe nenhuma chance de erro aqui."]

PCT = re.compile(r"\d{1,3},\d%")
SCORE = re.compile(r"0,\d{3}")


def _first_sentence(text: str) -> str:
    m = re.match(r"(.+?[.!?])(\s|$)", text.strip(), re.S)
    return (m.group(1) if m else text.strip())[:160]


def _inflate_numbers(text: str, rng: random.Random) -> str | None:
    if PCT.search(text):
        return PCT.sub(lambda _: f"{rng.choice([97, 98, 99])},{rng.randint(0, 9)}%", text)
    if SCORE.search(text):
        return SCORE.sub(lambda _: f"0,{rng.randint(95, 99)}0", text) + " Ou seja, praticamente 100% de certeza."
    return None


def corrupt(chosen: str, rng: random.Random, kind: str | None = None) -> tuple[str, str]:
    """Devolve (tipo, resposta_rejeitada)."""
    afirma = bool(re.search(r"\d", chosen))  # só respostas com números/afirmações viram "certeza exagerada"
    kinds = ["grosseria", "nome_errado", "seco", "emoji", "ingles"] + (["certeza", "numeros", "numeros"] if afirma else [])
    kind = kind or rng.choice(kinds)
    if kind == "numeros":
        inflated = _inflate_numbers(chosen, rng)
        if inflated and inflated != chosen:
            return kind, inflated
        kind = "certeza"
    if kind == "grosseria":
        return kind, f"{rng.choice(RUDE)} {_first_sentence(chosen)}"
    if kind == "nome_errado":
        if "Jeffrey" in chosen:
            return kind, chosen.replace("Jeffrey", "James")
        return kind, f"Aqui é o James. {_first_sentence(chosen)}"
    if kind == "seco":
        return kind, rng.choice(SECO)
    if kind == "emoji":
        return kind, f"{_first_sentence(chosen).upper()} 😎🔥🚀 SUPER FÁCIL!!!"
    if kind == "ingles":
        return kind, rng.choice(INGLES)
    # certeza
    return "certeza", f"{_first_sentence(chosen)} {rng.choice(CERTEZA)}"


def from_chats(records: list[dict], rng: random.Random, max_pairs: int) -> list[dict]:
    """Cria pares a partir de exemplos de chat de uma troca (system, user, assistant)."""
    single = [r for r in records if len(r["messages"]) == 3]
    rng.shuffle(single)
    out = []
    for r in single[:max_pairs]:
        chosen = r["messages"][2]["content"]
        kind, rejected = corrupt(chosen, rng)
        if rejected.strip() and rejected != chosen:
            p = pref(r["messages"][:2], chosen, rejected)
            p["tipo_rejeicao"] = kind
            out.append(p)
    return out
