"""Formatos, atalhos de fatos e validação dos datasets da persona Jeffrey."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.persona import NOME, PROMPT_SISTEMA, Fatos, fmt, carregar_fatos

F: Fatos = carregar_fatos()

# atalhos dos fatos, já formatados em pt-BR
T1, T5 = F.r("ajustado", "t2i_R@1"), F.r("ajustado", "t2i_R@5")
B1, B5 = F.r("base", "t2i_R@1"), F.r("base", "t2i_R@5")
I1B, I1T = F.r("base", "i2t_R@1"), F.r("ajustado", "i2t_R@1")
GANHO_T1 = F.ganho("t2i_R@1")
TAGS = ", ".join(F.tags)
N_IMG, N_LEG = fmt(F.n_imagens, 0), fmt(F.n_legendas, 0)
N_TREINO, N_VAL = fmt(F.n_treino, 0), fmt(F.n_val, 0)
MIN = fmt(F.minutos_treino)
CLIP_B, CLIP_T = fmt(F.clip_base_t2i_r1) + "%", fmt(F.clip_ajustado_t2i_r1) + "%"


# ---------------------------------------------------------------------------
# formatos
# ---------------------------------------------------------------------------
def chat(*turnos: str) -> dict:
    """chat("usuário", "assistente", "usuário", "assistente", ...) -> {"messages": [...]}"""
    msgs = [{"role": "system", "content": PROMPT_SISTEMA}]
    for i, t in enumerate(turnos):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
    return {"messages": msgs}


def variantes(pares: list[tuple[str | tuple[str, ...], str]]) -> list[dict]:
    """[(("pergunta", "outra forma"), "resposta")] -> um exemplo de chat por forma de perguntar."""
    exemplos = []
    for usuarios, resposta in pares:
        for u in (usuarios if isinstance(usuarios, tuple) else (usuarios,)):
            exemplos.append(chat(u, resposta))
    return exemplos


def instr(instrucao: str, entrada: str, saida: str) -> dict:
    return {"system": PROMPT_SISTEMA, "instruction": instrucao, "input": entrada, "output": saida}


def pref(usuario: str | list[dict], escolhida: str, rejeitada: str) -> dict:
    prompt = usuario if isinstance(usuario, list) else [{"role": "system", "content": PROMPT_SISTEMA},
                                                  {"role": "user", "content": usuario}]
    return {"prompt": prompt,
            "chosen": [{"role": "assistant", "content": escolhida}],
            "rejected": [{"role": "assistant", "content": rejeitada}]}


def instr_para_chat(r: dict) -> dict:
    usuario = r["instruction"] + (f"\n\n{r['input']}" if r["input"] else "")
    return chat(usuario, r["output"])


# ---------------------------------------------------------------------------
# validação
# ---------------------------------------------------------------------------
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
NOMES_ERRADOS = ("James", "James.IA")


def validar_texto_assistente(texto: str, onde: str) -> None:
    assert isinstance(texto, str) and texto.strip(), f"{onde}: resposta vazia"
    assert not any(n in texto for n in NOMES_ERRADOS), f"{onde}: nome errado: {texto[:70]}"
    assert not EMOJI.search(texto), f"{onde}: emoji: {texto[:70]}"
    assert "{" not in texto and "}" not in texto, f"{onde}: placeholder não resolvido: {texto[:70]}"


def validar_chat(reg: dict, onde: str) -> None:
    msgs = reg["messages"]
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == PROMPT_SISTEMA, f"{onde}: system ausente"
    papeis = [m["role"] for m in msgs[1:]]
    assert papeis and papeis[-1] == "assistant", f"{onde}: deve terminar com o assistente"
    assert all(r == ("user" if i % 2 == 0 else "assistant") for i, r in enumerate(papeis)), f"{onde}: ordem inválida"
    for m in msgs[1:]:
        assert isinstance(m["content"], str) and m["content"].strip(), f"{onde}: mensagem vazia"
        if m["role"] == "assistant":
            validar_texto_assistente(m["content"], onde)


def validar_pref(reg: dict, onde: str) -> None:
    validar_chat({"messages": reg["prompt"] + reg["chosen"]}, onde)
    assert reg["chosen"][0]["content"] != reg["rejected"][0]["content"], f"{onde}: chosen == rejected"
    assert reg["rejected"][0]["content"].strip(), f"{onde}: rejected vazio"


def validar_instr(reg: dict, onde: str) -> None:
    validar_chat(instr_para_chat(reg), onde)


def escrever_jsonl(caminho: Path, linhas: list[dict]) -> None:
    with open(caminho, "w", encoding="utf-8") as f:
        for r in linhas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


__all__ = [
    "F", "NOME", "PROMPT_SISTEMA", "fmt", "T1", "T5", "B1", "B5", "I1B", "I1T", "GANHO_T1", "TAGS", "N_IMG", "N_LEG",
    "N_TREINO", "N_VAL", "MIN", "CLIP_B", "CLIP_T", "chat", "variantes", "instr", "pref", "instr_para_chat",
    "validar_chat", "validar_pref", "validar_instr", "validar_texto_assistente", "escrever_jsonl",
]
