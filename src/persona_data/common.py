"""Formatos, atalhos de fatos e validação dos datasets da persona Jeffrey."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.persona import NAME, SYSTEM_PROMPT, Facts, fmt, load_facts

F: Facts = load_facts()

# atalhos dos fatos, já formatados em pt-BR
T1, T5 = F.r("tuned", "t2i_R@1"), F.r("tuned", "t2i_R@5")
B1, B5 = F.r("base", "t2i_R@1"), F.r("base", "t2i_R@5")
I1B, I1T = F.r("base", "i2t_R@1"), F.r("tuned", "i2t_R@1")
GAIN_T1 = F.gain("t2i_R@1")
TAGS = ", ".join(F.tags)
N_IMG, N_CAP = fmt(F.n_images, 0), fmt(F.n_captions, 0)
N_TRAIN, N_VAL = fmt(F.n_train, 0), fmt(F.n_val, 0)
MIN = fmt(F.train_minutes)
CLIP_B, CLIP_T = fmt(F.clip_base_t2i_r1) + "%", fmt(F.clip_tuned_t2i_r1) + "%"


# ---------------------------------------------------------------------------
# formatos
# ---------------------------------------------------------------------------
def chat(*turns: str) -> dict:
    """chat("usuário", "assistente", "usuário", "assistente", ...) -> {"messages": [...]}"""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i, t in enumerate(turns):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
    return {"messages": msgs}


def variants(pairs: list[tuple[str | tuple[str, ...], str]]) -> list[dict]:
    """[(("pergunta", "outra forma"), "resposta")] -> um exemplo de chat por forma de perguntar."""
    out = []
    for users, answer in pairs:
        for u in (users if isinstance(users, tuple) else (users,)):
            out.append(chat(u, answer))
    return out


def instr(instruction: str, input_: str, output: str) -> dict:
    return {"system": SYSTEM_PROMPT, "instruction": instruction, "input": input_, "output": output}


def pref(user: str | list[dict], chosen: str, rejected: str) -> dict:
    prompt = user if isinstance(user, list) else [{"role": "system", "content": SYSTEM_PROMPT},
                                                  {"role": "user", "content": user}]
    return {"prompt": prompt,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}]}


def instr_to_chat(r: dict) -> dict:
    user = r["instruction"] + (f"\n\n{r['input']}" if r["input"] else "")
    return chat(user, r["output"])


# ---------------------------------------------------------------------------
# validação
# ---------------------------------------------------------------------------
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
WRONG_NAMES = ("James", "James.IA")


def check_assistant_text(text: str, where: str) -> None:
    assert isinstance(text, str) and text.strip(), f"{where}: resposta vazia"
    assert not any(n in text for n in WRONG_NAMES), f"{where}: nome errado: {text[:70]}"
    assert not EMOJI.search(text), f"{where}: emoji: {text[:70]}"
    assert "{" not in text and "}" not in text, f"{where}: placeholder não resolvido: {text[:70]}"


def check_chat(rec: dict, where: str) -> None:
    msgs = rec["messages"]
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == SYSTEM_PROMPT, f"{where}: system ausente"
    roles = [m["role"] for m in msgs[1:]]
    assert roles and roles[-1] == "assistant", f"{where}: deve terminar com o assistente"
    assert all(r == ("user" if i % 2 == 0 else "assistant") for i, r in enumerate(roles)), f"{where}: ordem inválida"
    for m in msgs[1:]:
        assert isinstance(m["content"], str) and m["content"].strip(), f"{where}: mensagem vazia"
        if m["role"] == "assistant":
            check_assistant_text(m["content"], where)


def check_pref(rec: dict, where: str) -> None:
    check_chat({"messages": rec["prompt"] + rec["chosen"]}, where)
    assert rec["chosen"][0]["content"] != rec["rejected"][0]["content"], f"{where}: chosen == rejected"
    assert rec["rejected"][0]["content"].strip(), f"{where}: rejected vazio"


def check_instr(rec: dict, where: str) -> None:
    check_chat(instr_to_chat(rec), where)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


__all__ = [
    "F", "NAME", "SYSTEM_PROMPT", "fmt", "T1", "T5", "B1", "B5", "I1B", "I1T", "GAIN_T1", "TAGS", "N_IMG", "N_CAP",
    "N_TRAIN", "N_VAL", "MIN", "CLIP_B", "CLIP_T", "chat", "variants", "instr", "pref", "instr_to_chat",
    "check_chat", "check_pref", "check_instr", "check_assistant_text", "write_jsonl",
]
