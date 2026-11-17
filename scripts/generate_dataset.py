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
from src.persona import NOME  # noqa: E402
from src.persona_data import dpo, grounded, manual  # noqa: E402
from src.persona_data.common import (  # noqa: E402
    F, validar_chat, validar_instr, validar_pref, instr_para_chat, escrever_jsonl,
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


def dividir(exemplos: list, proporcao: float, rng: random.Random) -> tuple[list, list]:
    exemplos = exemplos[:]
    rng.shuffle(exemplos)
    n_val = max(1, round(len(exemplos) * proporcao)) if len(exemplos) >= 5 else 0
    return exemplos[n_val:], exemplos[:n_val]


def media_palavras(exemplos: list[dict]) -> float:
    tamanhos = []
    for r in exemplos:
        if "messages" in r:
            tamanhos += [len(m["content"].split()) for m in r["messages"] if m["role"] == "assistant"]
        elif "output" in r:
            tamanhos.append(len(r["output"].split()))
        else:
            tamanhos.append(len(r["chosen"][0]["content"].split()))
    return statistics.mean(tamanhos) if tamanhos else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(config.DIR_PROCESSADOS))
    ap.add_argument("--n-explicacoes", type=int, default=80)
    ap.add_argument("--n-imagem", type=int, default=50)
    ap.add_argument("--n-descricao", type=int, default=50)
    ap.add_argument("--n-comparacao", type=int, default=40)
    ap.add_argument("--n-conversas", type=int, default=40)
    ap.add_argument("--dpo-max", type=int, default=250, help="máximo de pares DPO gerados automaticamente")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--no-model", action="store_true", help="não carrega o modelo (imagem_para_texto sem zero-shot)")
    ap.add_argument("--seed", type=int, default=config.SEMENTE)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    dir_saida = Path(args.out)
    dir_saida.mkdir(parents=True, exist_ok=True)

    # ---- escritos à mão
    conjuntos_chat: dict[str, list[dict]] = {
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
    ajustado = grounded.carregar_indice("ptbr_finetuned")
    base = grounded.carregar_indice(re.sub(r"[^A-Za-z0-9]+", "_", config.MODELO_BASE).strip("_"))
    if ajustado is None:
        print("⚠️  Índice do modelo ajustado não encontrado (rode scripts/finetune.py e scripts/build_index.py). "
              "Gerando só os datasets escritos à mão.")
    else:
        nome = "SigLIP 2 ajustado em português"
        vetores_rotulos, escala = (None, None) if args.no_model else grounded.calcular_vetores_rotulos(str(config.DIR_MODELO_AJUSTADO))
        conjuntos_chat["explicacao_resultados.jsonl"] = grounded.explicacoes(ajustado, args.n_explicacoes, rng, nome)
        conjuntos_chat["imagem_para_imagem.jsonl"] = grounded.imagem_para_imagem(ajustado, args.n_imagem, rng, nome)
        conjuntos_chat["imagem_para_texto.jsonl"] = grounded.imagem_para_texto(ajustado, args.n_descricao, rng, vetores_rotulos, escala)
        conjuntos_chat["refinamento_busca.jsonl"] = grounded.refinamento(ajustado, rng)
        conjuntos_chat["catalogo.jsonl"] = grounded.catalogo(ajustado, rng)
        if base is not None:
            conjuntos_chat["comparacao_modelos.jsonl"] = grounded.comparacao(base, ajustado, args.n_comparacao, rng)
        conjuntos_chat["conversas_geradas.jsonl"] = grounded.conversas_geradas(ajustado, args.n_conversas, rng, nome)

    inst = manual.instrucoes()

    # ---- preferências: à mão + geradas a partir das respostas boas
    fontes_dpo = [r for k in ("identidade.jsonl", "conhecimento_projeto.jsonl", "limites.jsonl", "comportamento.jsonl",
                              "suporte_tecnico.jsonl", "uso_do_app.jsonl", "conceitos_ia.jsonl", "seguranca.jsonl", "catalogo.jsonl",
                              "explicacao_resultados.jsonl")
                  for r in conjuntos_chat.get(k, [])]
    prefs = manual.preferencias() + dpo.de_chats(fontes_dpo, rng, args.dpo_max)

    # ---- validação
    for nome, exemplos in conjuntos_chat.items():
        for i, r in enumerate(exemplos):
            validar_chat(r, f"{nome}#{i}")
    for i, r in enumerate(inst):
        validar_instr(r, f"treino_instrucao.jsonl#{i}")
    for i, r in enumerate(prefs):
        validar_pref(r, f"preferencias_dpo.jsonl#{i}")

    # ---- arquivos individuais
    estatisticas: list[dict] = []
    for nome, exemplos in conjuntos_chat.items():
        escrever_jsonl(dir_saida / nome, exemplos)
        estatisticas.append({"arquivo": nome, "formato": "chat", "exemplos": len(exemplos), "palavras_resposta": media_palavras(exemplos)})
    escrever_jsonl(dir_saida / "treino_instrucao.jsonl", inst)
    estatisticas.append({"arquivo": "treino_instrucao.jsonl", "formato": "instrução", "exemplos": len(inst),
                  "palavras_resposta": media_palavras(inst)})
    escrever_jsonl(dir_saida / "preferencias_dpo.jsonl", prefs)
    estatisticas.append({"arquivo": "preferencias_dpo.jsonl", "formato": "preferência", "exemplos": len(prefs),
                  "palavras_resposta": media_palavras(prefs)})

    # ---- combinados: divisão estratificada (cada arquivo contribui para treino e validação)
    sft_treino, sft_val, vistos = [], [], set()
    for exemplos in list(conjuntos_chat.values()) + [[instr_para_chat(x) for x in inst]]:
        unicas = []
        for r in exemplos:
            chave = json.dumps(r["messages"][1:], ensure_ascii=False)
            if chave not in vistos:
                vistos.add(chave)
                unicas.append(r)
        tr, va = dividir(unicas, args.val_ratio, rng)
        sft_treino += tr
        sft_val += va
    rng.shuffle(sft_treino)
    rng.shuffle(sft_val)
    dpo_treino, dpo_val = dividir(prefs, args.val_ratio, rng)
    for nome_arq, exemplos in (("jeffrey_sft_train.jsonl", sft_treino), ("jeffrey_sft_val.jsonl", sft_val),
                        ("jeffrey_dpo_train.jsonl", dpo_treino), ("jeffrey_dpo_val.jsonl", dpo_val)):
        escrever_jsonl(dir_saida / nome_arq, exemplos)
        estatisticas.append({"arquivo": nome_arq, "formato": "preferência" if "dpo" in nome_arq else "chat",
                      "exemplos": len(exemplos), "palavras_resposta": media_palavras(exemplos)})

    tipos: dict[str, int] = {}
    for p in prefs:
        k = p.get("tipo_rejeicao", "manual")
        tipos[k] = tipos.get(k, 0) + 1
    meta = {
        "persona": NOME, "seed": args.seed, "modelo_base": F.modelo_base,
        "metricas_citadas": {"t2i_R@1_base": F.base["t2i_R@1"], "t2i_R@1_ajustado": F.ajustado["t2i_R@1"]},
        "arquivos": estatisticas, "tipos_rejeicao_dpo": tipos,
    }
    (dir_saida / "stats.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    escrever_cartao(dir_saida, estatisticas, tipos)

    print(f"✅ Datasets da persona {NOME} gerados em {dir_saida}\n")
    print(f"{'arquivo':<32} {'formato':<12} {'exemplos':>8} {'palavras/resp':>14}")
    for s in estatisticas:
        print(f"{s['arquivo']:<32} {s['formato']:<12} {s['exemplos']:>8} {s['palavras_resposta']:>14.1f}")


def escrever_cartao(dir_saida: Path, estatisticas: list[dict], tipos: dict[str, int]) -> None:
    linhas = "\n".join(
        f"| `{s['arquivo']}` | {s['formato']} | {s['exemplos']} | {s['palavras_resposta']:.0f} | "
        f"{DESCRICOES.get(s['arquivo'], 'combinado (treino/validação)')} |"
        for s in estatisticas
    )
    tipos_txt = ", ".join(f"{k}: {v}" for k, v in sorted(tipos.items(), key=lambda kv: -kv[1]))
    cartao = f"""# Datasets da persona {NOME}

> Gerado por `python scripts/generate_dataset.py`. Não edite à mão: mude o gerador e rode de novo.

O **{NOME}** é o assistente do VisioVetor: cordial, educado e honesto. Ele só cita números que vêm do
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
    (dir_saida / "README.md").write_text(cartao, encoding="utf-8")


if __name__ == "__main__":
    main()
