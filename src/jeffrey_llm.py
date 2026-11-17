"""O Jeffrey com um modelo de linguagem de verdade (RAG em cima dos dados reais).

O LLM entende a pergunta; os FATOS vêm do sistema. A cada mensagem, o prompt leva:
  * os fatos do projeto (modelo, métricas, catálogo, faixas de score medidas no índice);
  * a última busca da aba (posições, scores, legendas, tags, passos, zero-shot);
  * a análise automática das regras (src/jeffrey_chat.py), quando houver;
  * as respostas mais parecidas dos datasets da persona (busca semântica com MiniLM).
Assim ele responde a pergunta que foi feita, mas só com números calculados pelo VisioVetor.

Configuração (variável de ambiente VISIOVETOR_JEFFREY_LLM):
  padrão  Qwen/Qwen2.5-3B-Instruct    (~6 GB; ~6 tokens/s em CPU com bf16)
  menor   Qwen/Qwen2.5-1.5B-Instruct  (~3 GB; 2x mais rápido, mas erra mais ao ler a lista de resultados)
  off     desliga o LLM; o chat volta a usar só as regras
"""

from __future__ import annotations

import copy
import os
import threading
from collections.abc import Iterator

import numpy as np
import torch

import config
from src.persona import PROMPT_SISTEMA, fmt

ID_LLM = os.getenv("VISIOVETOR_JEFFREY_LLM", "Qwen/Qwen2.5-3B-Instruct")
ID_CODIFICADOR_FRASES = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_TOKENS_NOVOS = 320
TURNOS_HISTORICO = 6

REGRAS = """
Regras para responder:
- Responda exatamente o que foi perguntado, logo na primeira frase. No total, no máximo 4 frases curtas.
- Use SOMENTE os fatos abaixo. Nunca invente números, legendas, imagens ou recursos. Se a resposta não estiver nos fatos, diga com franqueza que não sabe.
- Números no padrão brasileiro (0,175). Fale de "você". Sem emojis, sem markdown pesado.
- Você não roda buscas. Para buscar, a pessoa usa a caixa de busca no topo da aba; as imagens aparecem logo abaixo dela.
- Se o pedido fugir do VisioVetor (geografia, política, receitas, notícias...), NÃO responda o conteúdo: diga com gentileza que só ajuda com o VisioVetor.
- Quando a mensagem trouxer [Dados do sistema], eles estão corretos: baseie a resposta neles.
- Você não vê as fotos, só as legendas e os scores. Não afirme o que há numa foto além do que a legenda diz.""".strip()


# Exemplos de conversa (few-shot): modelos pequenos imitam o formato muito melhor do que seguem regras.
EXEMPLOS = [
    {"role": "user", "content": "qual o maior score que dá pra ter?"},
    {"role": "assistant", "content": "O máximo teórico é 1, que só acontece quando os dois vetores são idênticos. Na "
                                     "prática, entre texto e imagem os scores ficam bem abaixo disso, e tudo bem: o que "
                                     "importa é a ordem dos resultados."},
    {"role": "user", "content": "quem ganhou a última copa do mundo?"},
    {"role": "assistant", "content": "Essa eu vou ficar devendo: só consigo ajudar com o VisioVetor, como buscar imagens "
                                     "e explicar os resultados. Quer tentar uma busca?"},
    {"role": "user", "content": "qual dessas tem cachorro?"},
    {"role": "assistant", "content": "Pelas legendas, o cachorro aparece no 2º e no 5º lugar. Eu só vejo as legendas, "
                                     "então vale conferir as fotos."},
]


class CodificadorFrases:
    """MiniLM multilíngue (mean pooling), para achar perguntas parecidas por significado."""

    def __init__(self, id_modelo: str = ID_CODIFICADOR_FRASES):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(id_modelo)
        self.modelo = AutoModel.from_pretrained(id_modelo).eval()

    @torch.no_grad()
    def codificar(self, textos: list[str], lote: int = 64) -> np.ndarray:
        saida = []
        for i in range(0, len(textos), lote):
            lote_tok = self.tok(textos[i:i + lote], padding=True, truncation=True, max_length=128, return_tensors="pt")
            h = self.modelo(**lote_tok).last_hidden_state
            mascara = lote_tok["attention_mask"].unsqueeze(-1).float()
            v = (h * mascara).sum(1) / mascara.sum(1)
            saida.append(torch.nn.functional.normalize(v, dim=-1).numpy())
        return np.concatenate(saida) if saida else np.zeros((0, 384), dtype=np.float32)


class JeffreyLLM:
    def __init__(self, regras, fatos, estatisticas_score: dict | None = None):
        self.regras = regras            # JeffreyChat (regras + base de conhecimento)
        self.fatos = fatos
        self.estatisticas_score = estatisticas_score or {}
        self.ativo = ID_LLM.lower() not in {"off", "0", "false", "none", ""}
        self.pronto = threading.Event()
        self.erro: str | None = None
        self._trava = threading.Lock()
        self.modelo = self.tok = self.codificador_frases = self.vetores_bc = None
        self.ids_prefixo = self.cache_prefixo = None  # começo fixo do prompt, já processado

    # -- carga em segundo plano (o app abre sem esperar) ---------------------------
    def carregar_em_segundo_plano(self) -> None:
        if self.ativo:
            threading.Thread(target=self._carregar, daemon=True).start()

    def _carregar(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.codificador_frases = CodificadorFrases()
            self.vetores_bc = self.codificador_frases.codificar(self.regras.bc.perguntas)
            dtype = torch.float16 if config.DISPOSITIVO == "cuda" else torch.bfloat16
            self.tok = AutoTokenizer.from_pretrained(ID_LLM)
            self.modelo = AutoModelForCausalLM.from_pretrained(ID_LLM, dtype=dtype).to(config.DISPOSITIVO).eval()
            torch.set_num_threads(max(1, os.cpu_count() or 1))
            self._montar_cache_prefixo()
            print(f"🤵 Jeffrey com LLM pronto: {ID_LLM}")
        except Exception as e:  # sem internet, sem memória... o chat segue com as regras
            self.erro = f"{type(e).__name__}: {e}"
            print(f"⚠️  LLM do Jeffrey indisponível ({self.erro}); usando só as regras.")
        finally:
            self.pronto.set()

    def _mensagens_fixas(self) -> list[dict]:
        """Parte do prompt que não muda durante a execução: persona, regras, fatos e exemplos."""
        sistema = "\n\n".join([PROMPT_SISTEMA, REGRAS, "FATOS DO PROJETO:\n" + self._fatos_projeto(),
                                "A cada pergunta, a mensagem traz os dados da última busca da aba e referências."])
        return [{"role": "system", "content": sistema}] + EXEMPLOS

    @torch.no_grad()
    def _montar_cache_prefixo(self) -> None:
        """Processa o começo fixo uma vez só; cada pergunta só paga pelo que é novo (~3x mais rápido em CPU)."""
        try:
            from transformers import DynamicCache
            ids = self.tok.apply_chat_template(self._mensagens_fixas(), return_tensors="pt",
                                               return_dict=True)["input_ids"].to(self.modelo.device)
            cache = DynamicCache()
            self.modelo(input_ids=ids, past_key_values=cache, use_cache=True)
            self.ids_prefixo, self.cache_prefixo = ids, cache
        except Exception as e:
            print(f"⚠️  Sem cache de prefixo ({type(e).__name__}: {e}); o Jeffrey só fica mais lento.")

    @property
    def disponivel(self) -> bool:
        return self.ativo and self.pronto.is_set() and self.modelo is not None

    # -- montagem do contexto ------------------------------------------------------------
    def _fatos_projeto(self) -> str:
        f, r = self.fatos, self.regras
        tags = ", ".join(f"{t} ({n})" for t, n in r.contagem_tags.most_common())
        linhas = [
            f"- Modelo de busca: {f.modelo_base} (SigLIP 2, multilíngue), ajustado em português; vetores de {f.dim} "
            "dimensões; imagens e textos no mesmo espaço.",
            f"- Catálogo: {fmt(len(r.registros), 0)} imagens do Flickr8k com {fmt(sum(len(x['captions']) for x in r.registros), 0)} "
            "legendas em português (cerca de 5 por imagem).",
            f"- Tags automáticas (contagem): {tags}. Com várias tags no filtro, a imagem precisa ter todas.",
            f"- Fine-tuning: partimos do SigLIP 2 já treinado pelo Google. O encoder de imagem ficou congelado; só as "
            f"últimas {f.camadas_texto} camadas do encoder de texto foram ajustadas, com pares (foto, legenda em português) "
            f"de {fmt(f.n_treino, 0)} imagens de treino, {f.epocas} épocas, cerca de {fmt(f.minutos_treino)} min em CPU, "
            "perda sigmoide. Objetivo: entender melhor frases em português. Não houve CLIP nem destilação nesse treino.",
            f"- Cada busca compara a consulta com TODAS as {fmt(len(r.registros), 0)} imagens do catálogo (as de treino "
            "também aparecem nas buscas).",
            f"- Validação ({fmt(f.n_val, 0)} imagens fora do treino), texto→imagem: R@1 base {f.r('base', 't2i_R@1')}, "
            f"ajustado {f.r('ajustado', 't2i_R@1')}; R@5 base {f.r('base', 't2i_R@5')}, ajustado {f.r('ajustado', 't2i_R@5')}. "
            f"Imagem→texto R@1: base {f.r('base', 'i2t_R@1')}, ajustado {f.r('ajustado', 'i2t_R@1')}. "
            "R@K = % das buscas em que a imagem certa aparece entre as K primeiras.",
            f"- O CLIP original da OpenAI, testado antes, fazia R@1 {fmt(f.clip_base_t2i_r1)}% em português "
            f"({fmt(f.clip_ajustado_t2i_r1)}% ajustado).",
            "- IMPORTANTE: na busca texto→imagem, a frase é comparada com a FOTO (pixels), não com a legenda. A "
            "legenda é só para a pessoa ler; por isso o ranking pode não seguir as palavras das legendas, e uma foto "
            "pode ter coisas que a legenda não cita.",
            "- Score = similaridade de cosseno entre vetores normalizados. LIMITES TEÓRICOS: máximo 1 (vetores "
            "idênticos) e mínimo -1 (vetores opostos). Não é porcentagem de certeza: serve para ordenar.",
        ]
        s = self.estatisticas_score
        if s:
            linhas.append(
                f"- Medido no índice atual: texto→imagem, o 1º lugar costuma ficar entre {fmt(s['t2i_p5'], 3)} e "
                f"{fmt(s['t2i_p95'], 3)} (mediana {fmt(s['t2i_med'], 3)}); o maior score texto→imagem visto foi "
                f"{fmt(s['t2i_max'], 3)}. Considerando todos os pares texto-imagem, inclusive os que não combinam, "
                f"os valores OBSERVADOS na prática vão de {fmt(s['all_min'], 3)} a {fmt(s['t2i_max'], 3)} (metade fica abaixo de "
                f"{fmt(s['all_med'], 3)}): valores negativos existem, mas só para pares sem relação. "
                f"Entre imagens, a mais parecida costuma ter {fmt(s['i2i_med'], 3)}. "
                "Por isso, no SigLIP, scores de texto→imagem baixos (0,1 a 0,3) são normais; scores de modelos "
                "diferentes não são comparáveis.")
        linhas.append("- Zero-shot (aba Imagem → Texto): compara a foto com \"uma foto de <rótulo>\"; as "
                      "porcentagens são relativas aos rótulos digitados e somam 100%.")
        linhas.append("- O app: abas Texto → Imagem, Imagem → Imagem, Imagem → Texto e Base × Ajustado; roda local "
                      "(Gradio), sem enviar dados para fora. Você, Jeffrey, não vê as imagens: só legendas e scores.")
        return "\n".join(linhas)

    @staticmethod
    def _fatos_busca(ctx: dict | None) -> str:
        if not ctx or not ctx.get("results"):
            return "Nenhuma busca feita nesta aba ainda."
        tipo = ctx["kind"]
        nome = {"texto": "texto → imagem", "imagem": "imagem → imagem (a pessoa enviou uma foto)",
                "descrever": "imagem → texto (legendas do catálogo mais próximas da foto enviada)",
                "comparar": "comparação base × ajustado (lista abaixo = modelo ajustado)"}[tipo]
        linhas = [f"Tipo: {nome}."]
        if ctx.get("query"):
            linhas.append(f"Busca digitada: \"{ctx['query']}\".")
        if ctx.get("tags"):
            linhas.append(f"Filtro de tags: {', '.join(ctx['tags'])}.")
        sc = [r["score"] for r in ctx["results"]]
        linhas.append(f"Ordem: o 1º tem o MAIOR score ({fmt(max(sc), 3)}); o último, o menor ({fmt(min(sc), 3)}). "
                      + ("Todos os scores desta lista são positivos." if min(sc) > 0 else ""))
        for i, r in enumerate(ctx["results"], 1):
            tags = f" [tags: {', '.join(r['tags'])}]" if r.get("tags") else ""
            linhas.append(f"{i}º: score {fmt(r['score'], 3)}, legenda \"{r['caption'].rstrip('.')}\"{tags}")
        if ctx.get("base"):
            linhas.append("Modelo base, mesma busca:")
            linhas += [f"{i}º: score {fmt(b['score'], 3)}, \"{b['caption'].rstrip('.')}\""
                       for i, b in enumerate(ctx["base"], 1)]
        if ctx.get("zero_shot"):
            zs = sorted(ctx["zero_shot"].items(), key=lambda kv: -kv[1])
            linhas.append("Zero-shot: " + ", ".join(f"{k} {fmt(100 * v, 0)}%" for k, v in zs))
        if ctx.get("steps"):
            linhas.append("O que você fez: " + " ".join(ctx["steps"]))
        return "\n".join(linhas)

    def _referencias(self, msg: str, k: int = 2) -> list[tuple[str, str]]:
        if self.codificador_frases is None or self.vetores_bc is None or not len(self.vetores_bc):
            return []
        v = self.codificador_frases.codificar([msg])[0]
        sims = self.vetores_bc @ v
        idx = [i for i in np.argsort(-sims)[:k] if sims[i] >= 0.45]
        return [(self.regras.bc.perguntas[i], self.regras.bc.respostas[i]) for i in idx]

    def montar_mensagens(self, msg: str, ctx: dict | None, historico: list[dict], tipo: str) -> list[dict]:
        msgs = self._mensagens_fixas()
        for m in (historico or [])[-2 * TURNOS_HISTORICO:]:
            conteudo = m.get("content")
            if isinstance(conteudo, list):  # formato do Gradio 6: [{"type": "text", "text": ...}]
                conteudo = " ".join(c.get("text", "") for c in conteudo if isinstance(c, dict))
            if m.get("role") in ("user", "assistant") and isinstance(conteudo, str) and conteudo:
                msgs.append({"role": m["role"], "content": conteudo})
        # tudo o que muda vai junto da pergunta: fica fora do cache e perto de onde o modelo presta mais atenção
        extra = ["[Última busca nesta aba]\n" + self._fatos_busca(ctx)]
        if (refs := self._referencias(msg)):
            extra.append("[Referências: perguntas parecidas já respondidas; use só se servirem]\n" +
                         "\n".join(f"P: {q}\nR: {a}" for q, a in refs))
        if (analise := self.regras.analise(msg, ctx)):
            extra.append(f"[Dados do sistema para esta pergunta, corretos (use-os na resposta)]\n{analise}")
        extra.append("[Lembrete: só responda se for sobre o VisioVetor, imagens, buscas ou IA; se for outro assunto, "
                     "como geografia, esportes ou receitas, recuse com gentileza sem dar a resposta.]")
        msgs.append({"role": "user", "content": "\n\n".join(extra) + f"\n\nPergunta: {msg}"})
        return msgs

    # -- geração com streaming -------------------------------------------------------------------
    def gerar_em_fluxo(self, msg: str, ctx: dict | None, historico: list[dict], tipo: str) -> Iterator[str]:
        from transformers import TextIteratorStreamer
        msgs = self.montar_mensagens(msg, ctx, historico, tipo)
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to(self.modelo.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        n = 0 if self.ids_prefixo is None else self.ids_prefixo.shape[1]
        if n and torch.equal(ids["input_ids"][0, :n], self.ids_prefixo[0]):
            ids["past_key_values"] = copy.deepcopy(self.cache_prefixo)  # reaproveita o começo fixo
        kwargs = dict(**ids, streamer=streamer, max_new_tokens=MAX_TOKENS_NOVOS, do_sample=False,
                      repetition_penalty=1.05, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        with self._trava:  # um texto por vez: a CPU já usa todos os núcleos
            t = threading.Thread(target=lambda: self.modelo.generate(**kwargs), daemon=True)
            t.start()
            texto = ""
            for pedaco in streamer:
                texto += pedaco
                yield texto.strip()
            t.join()


def estatisticas_score(imagens, legendas, amostra: int = 300, semente: int = config.SEMENTE) -> dict:
    """Faixas reais de score no índice (para o Jeffrey não chutar)."""
    rng = np.random.default_rng(semente)
    ci = rng.choice(len(legendas.vetores), size=min(amostra, len(legendas.vetores)), replace=False)
    t2i = (legendas.vetores[ci] @ imagens.vetores.T).max(1)  # melhor imagem para cada legenda
    ii = rng.choice(len(imagens.vetores), size=min(amostra, len(imagens.vetores)), replace=False)
    sims = imagens.vetores[ii] @ imagens.vetores.T
    sims[np.arange(len(ii)), ii] = -1  # ignora a própria imagem
    todos = legendas.vetores[ci] @ imagens.vetores.T
    return {"t2i_p5": float(np.percentile(t2i, 5)), "t2i_med": float(np.median(t2i)),
            "t2i_p95": float(np.percentile(t2i, 95)), "t2i_max": float(todos.max()),
            "all_min": float(todos.min()), "all_med": float(np.median(todos)),
            "i2i_med": float(np.median(sims.max(1)))}
