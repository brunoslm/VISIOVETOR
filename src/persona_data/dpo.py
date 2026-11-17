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

GROSSEIRO = ["Isso é óbvio.", "Leia a documentação.", "Já expliquei isso antes.", "Pergunta básica, hein.",
        "Não tenho tempo pra isso.", "Se vira."]
SECO = ["ok", "sim", "não sei", "depende", "pesquisa aí", "tanto faz"]
INGLES = ["Sure! Just type what you want and click Search.", "I don't know, check the docs.",
          "That's how the model works. Anything else?"]
CERTEZA = ["Com certeza absoluta, o sistema nunca erra nisso.", "Garanto 100%, pode confiar cegamente.",
           "Não existe nenhuma chance de erro aqui."]

PCT = re.compile(r"\d{1,3},\d%")
SCORE = re.compile(r"0,\d{3}")


def _primeira_frase(texto: str) -> str:
    m = re.match(r"(.+?[.!?])(\s|$)", texto.strip(), re.S)
    return (m.group(1) if m else texto.strip())[:160]


def _inflar_numeros(texto: str, rng: random.Random) -> str | None:
    if PCT.search(texto):
        return PCT.sub(lambda _: f"{rng.choice([97, 98, 99])},{rng.randint(0, 9)}%", texto)
    if SCORE.search(texto):
        return SCORE.sub(lambda _: f"0,{rng.randint(95, 99)}0", texto) + " Ou seja, praticamente 100% de certeza."
    return None


def corromper(escolhida: str, rng: random.Random, tipo: str | None = None) -> tuple[str, str]:
    """Devolve (tipo, resposta_rejeitada)."""
    afirma = bool(re.search(r"\d", escolhida))  # só respostas com números/afirmações viram "certeza exagerada"
    tipos_possiveis = ["grosseria", "nome_errado", "seco", "emoji", "ingles"] + (["certeza", "numeros", "numeros"] if afirma else [])
    tipo = tipo or rng.choice(tipos_possiveis)
    if tipo == "numeros":
        inflado = _inflar_numeros(escolhida, rng)
        if inflado and inflado != escolhida:
            return tipo, inflado
        tipo = "certeza"
    if tipo == "grosseria":
        return tipo, f"{rng.choice(GROSSEIRO)} {_primeira_frase(escolhida)}"
    if tipo == "nome_errado":
        if "Jeffrey" in escolhida:
            return tipo, escolhida.replace("Jeffrey", "James")
        return tipo, f"Aqui é o James. {_primeira_frase(escolhida)}"
    if tipo == "seco":
        return tipo, rng.choice(SECO)
    if tipo == "emoji":
        return tipo, f"{_primeira_frase(escolhida).upper()} 😎🔥🚀 SUPER FÁCIL!!!"
    if tipo == "ingles":
        return tipo, rng.choice(INGLES)
    # certeza
    return "certeza", f"{_primeira_frase(escolhida)} {rng.choice(CERTEZA)}"


def de_chats(registros: list[dict], rng: random.Random, max_pares: int) -> list[dict]:
    """Cria pares a partir de exemplos de chat de uma troca (system, user, assistant)."""
    unicos = [r for r in registros if len(r["messages"]) == 3]
    rng.shuffle(unicos)
    saida = []
    for r in unicos[:max_pares]:
        escolhida = r["messages"][2]["content"]
        tipo, rejeitada = corromper(escolhida, rng)
        if rejeitada.strip() and rejeitada != escolhida:
            p = pref(r["messages"][:2], escolhida, rejeitada)
            p["tipo_rejeicao"] = tipo
            saida.append(p)
    return saida
