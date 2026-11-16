"""Gera os datasets JSONL de alinhamento da persona "Jeffrey".

O Jeffrey é o assistente do VisioVetor: cordial, educado e honesto. Estes arquivos
servem para ajustar (SFT e DPO) um LLM pequeno que converse sobre as buscas. Os
fatos citados vêm do projeto real (metrics.json, catálogo e índices de vetores),
então basta rodar de novo depois de re-treinar o modelo ou trocar o catálogo.

Conteúdo (data/processed/):
  escritos à mão  identidade, conhecimento_projeto, limites, comportamento, treino_chat,
                  suporte_tecnico, uso_do_app, conceitos_ia, seguranca, treino_instrucao
  dados reais     explicacao_resultados, imagem_para_imagem, imagem_para_texto,
                  refinamento_busca, catalogo, comparacao_modelos, conversas_geradas
  preferências    preferencias_dpo (à mão + geradas a partir das respostas boas)
  combinados      jeffrey_sft_{train,val}.jsonl, jeffrey_dpo_{train,val}.jsonl
  documentação    README.md (cartão do dataset) e stats.json

Uso:
  python scripts/generate_dataset.py
  python scripts/generate_dataset.py --n-explicacoes 150 --dpo-max 400
  python scripts/generate_dataset.py --no-model     # não carrega o modelo (sem zero-shot)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.persona import NAME  # noqa: E402
from src.persona_data import dpo, grounded, manual  # noqa: E402
from src.persona_data.common import (  # noqa: E402
    F, check_chat, check_instr, check_pref, instr_to_chat, write_jsonl,
)

DESCRICOES = {
    "identidade.jsonl": "nome, saudações, despedidas, tom e como lidar com grosseria (várias formas de perguntar)",
    "conhecimento_projeto.jsonl": "perguntas e respostas sobre o VisioVetor: modelo, métricas, score, tags, uso",
    "limites.jsonl": "recusas gentis: privacidade, saúde, métricas inventadas, pedidos fora do escopo",
    "comportamento.jsonl": "casos de comportamento originais do projeto, na voz do Jeffrey",
    "treino_chat.jsonl": "conversas com várias trocas escritas à mão (suporte, dúvidas, refinamento)",
    "suporte_tecnico.jsonl": "erros comuns de instalação e uso, com a solução",
    "uso_do_app.jsonl": "como usar o app e o chat do Jeffrey: o que perguntar, buscar pelo chat, treino, fotos próprias",
    "conceitos_ia.jsonl": "conceitos de IA explicados pelo Jeffrey (embedding, cosseno, LoRA, DPO...)",
    "seguranca.jsonl": "tentativas de manipulação, injeção de instruções e buscas preconceituosas",
    "explicacao_resultados.jsonl": "explica buscas REAIS de texto→imagem (scores verdadeiros, às vezes com os passos)",
    "imagem_para_imagem.jsonl": "explica buscas REAIS de imagens parecidas",
    "imagem_para_texto.jsonl": "explica legendas recuperadas e zero-shot REAIS para uma foto",
    "refinamento_busca.jsonl": "busca vaga → sugestões tiradas das legendas reais do catálogo",
    "catalogo.jsonl": "contagens reais por tag e \"tem foto de X?\" respondido com honestidade",
    "comparacao_modelos.jsonl": "compara resultados REAIS do modelo base e do ajustado",
    "conversas_geradas.jsonl": "saudação → busca real explicada → refinamento → despedida",
    "treino_instrucao.jsonl": "formato instruction / input / output (estilo Alpaca)",
    "preferencias_dpo.jsonl": "prompt / chosen / rejected (TRL), à mão + rejeições geradas",
}


def split(rows: list, ratio: float, rng: random.Random) -> tuple[list, list]:
    rows = rows[:]
    rng.shuffle(rows)
    n_val = max(1, round(len(rows) * ratio)) if len(rows) >= 5 else 0
    return rows[n_val:], rows[:n_val]


def avg_words(rows: list[dict]) -> float:
    lens = []
    for r in rows:
        if "messages" in r:
            lens += [len(m["content"].split()) for m in r["messages"] if m["role"] == "assistant"]
        elif "output" in r:
            lens.append(len(r["output"].split()))
        else:
            lens.append(len(r["chosen"][0]["content"].split()))
    return statistics.mean(lens) if lens else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(config.PROCESSED_DIR))
    ap.add_argument("--n-explicacoes", type=int, default=80)
    ap.add_argument("--n-imagem", type=int, default=50)
    ap.add_argument("--n-descricao", type=int, default=50)
    ap.add_argument("--n-comparacao", type=int, default=40)
    ap.add_argument("--n-conversas", type=int, default=40)
    ap.add_argument("--dpo-max", type=int, default=250, help="máximo de pares DPO gerados automaticamente")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--no-model", action="store_true", help="não carrega o modelo (imagem_para_texto sem zero-shot)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- escritos à mão
    chat_sets: dict[str, list[dict]] = {
        "identidade.jsonl": manual.identidade(),
        "conhecimento_projeto.jsonl": manual.conhecimento(),
        "limites.jsonl": manual.limites(),
        "comportamento.jsonl": manual.comportamento(),
        "treino_chat.jsonl": manual.conversas(),
        "suporte_tecnico.jsonl": manual.suporte(),
        "uso_do_app.jsonl": manual.uso_do_app(),
        "conceitos_ia.jsonl": manual.conceitos(),
        "seguranca.jsonl": manual.seguranca(),
    }

    # ---- ancorados nos dados reais
    tuned = grounded.load_index("ptbr_finetuned")
    base = grounded.load_index(re.sub(r"[^A-Za-z0-9]+", "_", config.BASE_MODEL).strip("_"))
    if tuned is None:
        print("⚠️  Índice do modelo ajustado não encontrado (rode scripts/finetune.py e scripts/build_index.py). "
              "Gerando só os datasets escritos à mão.")
    else:
        name = "SigLIP 2 ajustado em português"
        label_vecs, scale = (None, None) if args.no_model else grounded.label_vectors(str(config.FINETUNED_MODEL_DIR))
        chat_sets["explicacao_resultados.jsonl"] = grounded.explicacoes(tuned, args.n_explicacoes, rng, name)
        chat_sets["imagem_para_imagem.jsonl"] = grounded.imagem_para_imagem(tuned, args.n_imagem, rng, name)
        chat_sets["imagem_para_texto.jsonl"] = grounded.imagem_para_texto(tuned, args.n_descricao, rng, label_vecs, scale)
        chat_sets["refinamento_busca.jsonl"] = grounded.refinamento(tuned, rng)
        chat_sets["catalogo.jsonl"] = grounded.catalogo(tuned, rng)
        if base is not None:
            chat_sets["comparacao_modelos.jsonl"] = grounded.comparacao(base, tuned, args.n_comparacao, rng)
        chat_sets["conversas_geradas.jsonl"] = grounded.conversas_geradas(tuned, args.n_conversas, rng, name)

    inst = manual.instrucoes()

    # ---- preferências: à mão + geradas a partir das respostas boas
    fontes_dpo = [r for k in ("identidade.jsonl", "conhecimento_projeto.jsonl", "limites.jsonl", "comportamento.jsonl",
                              "suporte_tecnico.jsonl", "uso_do_app.jsonl", "conceitos_ia.jsonl", "seguranca.jsonl", "catalogo.jsonl",
                              "explicacao_resultados.jsonl")
                  for r in chat_sets.get(k, [])]
    prefs = manual.preferencias() + dpo.from_chats(fontes_dpo, rng, args.dpo_max)

    # ---- validação
    for name, rows in chat_sets.items():
        for i, r in enumerate(rows):
            check_chat(r, f"{name}#{i}")
    for i, r in enumerate(inst):
        check_instr(r, f"treino_instrucao.jsonl#{i}")
    for i, r in enumerate(prefs):
        check_pref(r, f"preferencias_dpo.jsonl#{i}")

    # ---- arquivos individuais
    stats: list[dict] = []
    for name, rows in chat_sets.items():
        write_jsonl(out / name, rows)
        stats.append({"arquivo": name, "formato": "chat", "exemplos": len(rows), "palavras_resposta": avg_words(rows)})
    write_jsonl(out / "treino_instrucao.jsonl", inst)
    stats.append({"arquivo": "treino_instrucao.jsonl", "formato": "instrução", "exemplos": len(inst),
                  "palavras_resposta": avg_words(inst)})
    write_jsonl(out / "preferencias_dpo.jsonl", prefs)
    stats.append({"arquivo": "preferencias_dpo.jsonl", "formato": "preferência", "exemplos": len(prefs),
                  "palavras_resposta": avg_words(prefs)})

    # ---- combinados: divisão estratificada (cada arquivo contribui para treino e validação)
    sft_train, sft_val, seen = [], [], set()
    for rows in list(chat_sets.values()) + [[instr_to_chat(x) for x in inst]]:
        uniq = []
        for r in rows:
            key = json.dumps(r["messages"][1:], ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                uniq.append(r)
        tr, va = split(uniq, args.val_ratio, rng)
        sft_train += tr
        sft_val += va
    rng.shuffle(sft_train)
    rng.shuffle(sft_val)
    dpo_train, dpo_val = split(prefs, args.val_ratio, rng)
    for fname, rows in (("jeffrey_sft_train.jsonl", sft_train), ("jeffrey_sft_val.jsonl", sft_val),
                        ("jeffrey_dpo_train.jsonl", dpo_train), ("jeffrey_dpo_val.jsonl", dpo_val)):
        write_jsonl(out / fname, rows)
        stats.append({"arquivo": fname, "formato": "preferência" if "dpo" in fname else "chat",
                      "exemplos": len(rows), "palavras_resposta": avg_words(rows)})

    tipos: dict[str, int] = {}
    for p in prefs:
        k = p.get("tipo_rejeicao", "manual")
        tipos[k] = tipos.get(k, 0) + 1
    meta = {
        "persona": NAME, "seed": args.seed, "modelo_base": F.base_model,
        "metricas_citadas": {"t2i_R@1_base": F.base["t2i_R@1"], "t2i_R@1_ajustado": F.tuned["t2i_R@1"]},
        "arquivos": stats, "tipos_rejeicao_dpo": tipos,
    }
    (out / "stats.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_card(out, stats, tipos)

    print(f"✅ Datasets da persona {NAME} gerados em {out}\n")
    print(f"{'arquivo':<32} {'formato':<12} {'exemplos':>8} {'palavras/resp':>14}")
    for s in stats:
        print(f"{s['arquivo']:<32} {s['formato']:<12} {s['exemplos']:>8} {s['palavras_resposta']:>14.1f}")


def write_card(out: Path, stats: list[dict], tipos: dict[str, int]) -> None:
    linhas = "\n".join(
        f"| `{s['arquivo']}` | {s['formato']} | {s['exemplos']} | {s['palavras_resposta']:.0f} | "
        f"{DESCRICOES.get(s['arquivo'], 'combinado (treino/validação)')} |"
        for s in stats
    )
    tipos_txt = ", ".join(f"{k}: {v}" for k, v in sorted(tipos.items(), key=lambda kv: -kv[1]))
    card = f"""# Datasets da persona {NAME}

> Gerado por `python scripts/generate_dataset.py`. Não edite à mão: mude o gerador e rode de novo.

O **{NAME}** é o assistente do VisioVetor: cordial, educado e honesto. Ele só cita números que vêm do
sistema, admite quando não sabe e recusa com gentileza o que foge do escopo.

| Arquivo | Formato | Exemplos | Palavras por resposta | Conteúdo |
|---|---|---:|---:|---|
{linhas}

## Formatos

- **chat**: `{{"messages": [{{"role": "system"|"user"|"assistant", "content": "..."}}]}}`, compatível com o
  `SFTTrainer` do TRL, Axolotl e LLaMA-Factory.
- **instrução**: `{{"system", "instruction", "input", "output"}}` (estilo Alpaca).
- **preferência**: `{{"prompt": [...], "chosen": [...], "rejected": [...]}}`, compatível com o `DPOTrainer` do TRL.
  O campo extra `tipo_rejeicao` diz como a resposta rejeitada quebra a persona ({tipos_txt}).

## Como os dados foram feitos

- **À mão**: identidade, conhecimento, limites, suporte, conceitos, segurança, instruções e conversas.
  Muitas perguntas aparecem em várias formas, todas com a mesma resposta.
- **A partir dos dados reais**: buscas feitas nos índices de vetores do projeto, com os scores
  verdadeiros. As consultas são legendas das imagens de **validação**. As explicações usam o mesmo
  narrador do painel do app (`src/jeffrey.py`).
- **Preferências**: pares escritos à mão + rejeições geradas automaticamente a partir das boas respostas.
- **Validação automática**: ordem das mensagens, respostas vazias, nome errado, emojis e placeholders.
- **Combinados**: cada arquivo é dividido em treino/validação separadamente (estratificado) e as
  duplicatas são removidas.
"""
    (out / "README.md").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    main()
