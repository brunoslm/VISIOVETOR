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
from src.persona import SYSTEM_PROMPT, fmt

LLM_ID = os.getenv("VISIOVETOR_JEFFREY_LLM", "Qwen/Qwen2.5-3B-Instruct")
EMBED_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_NEW_TOKENS = 320
HISTORY_TURNS = 6

REGRAS = """
Regras para responder:
- Responda exatamente o que foi perguntado, logo na primeira frase. No total, no máximo 4 frases curtas.
- Use SOMENTE os fatos abaixo. Nunca invente números, legendas, imagens ou recursos. Se a resposta não estiver nos fatos, diga com franqueza que não sabe.
- Números no padrão brasileiro (0,175). Fale de "você". Sem emojis, sem markdown pesado.
- Para buscar imagens, a pessoa pode escrever "busca" seguido da cena; você não roda buscas sozinho.
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


class SentenceEmbedder:
    """MiniLM multilíngue (mean pooling), para achar perguntas parecidas por significado."""

    def __init__(self, model_id: str = EMBED_ID):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval()

    @torch.no_grad()
    def encode(self, texts: list[str], batch: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=128, return_tensors="pt")
            h = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(v, dim=-1).numpy())
        return np.concatenate(out) if out else np.zeros((0, 384), dtype=np.float32)


class JeffreyLLM:
    def __init__(self, rules, facts, score_stats: dict | None = None):
        self.rules = rules            # JeffreyChat (regras + base de conhecimento)
        self.facts = facts
        self.score_stats = score_stats or {}
        self.enabled = LLM_ID.lower() not in {"off", "0", "false", "none", ""}
        self.ready = threading.Event()
        self.error: str | None = None
        self._lock = threading.Lock()
        self.model = self.tok = self.embedder = self.kb_vecs = None
        self.prefix_ids = self.prefix_cache = None  # começo fixo do prompt, já processado

    # -- carga em segundo plano (o app abre sem esperar) ---------------------------
    def load_async(self) -> None:
        if self.enabled:
            threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.embedder = SentenceEmbedder()
            self.kb_vecs = self.embedder.encode(self.rules.kb.questions)
            dtype = torch.float16 if config.DEVICE == "cuda" else torch.bfloat16
            self.tok = AutoTokenizer.from_pretrained(LLM_ID)
            self.model = AutoModelForCausalLM.from_pretrained(LLM_ID, dtype=dtype).to(config.DEVICE).eval()
            torch.set_num_threads(max(1, os.cpu_count() or 1))
            self._build_prefix_cache()
            print(f"🤵 Jeffrey com LLM pronto: {LLM_ID}")
        except Exception as e:  # sem internet, sem memória... o chat segue com as regras
            self.error = f"{type(e).__name__}: {e}"
            print(f"⚠️  LLM do Jeffrey indisponível ({self.error}); usando só as regras.")
        finally:
            self.ready.set()

    def _static_messages(self) -> list[dict]:
        """Parte do prompt que não muda durante a execução: persona, regras, fatos e exemplos."""
        system = "\n\n".join([SYSTEM_PROMPT, REGRAS, "FATOS DO PROJETO:\n" + self._project_facts(),
                                "A cada pergunta, a mensagem traz os dados da última busca da aba e referências."])
        return [{"role": "system", "content": system}] + EXEMPLOS

    @torch.no_grad()
    def _build_prefix_cache(self) -> None:
        """Processa o começo fixo uma vez só; cada pergunta só paga pelo que é novo (~3x mais rápido em CPU)."""
        try:
            from transformers import DynamicCache
            ids = self.tok.apply_chat_template(self._static_messages(), return_tensors="pt",
                                               return_dict=True)["input_ids"].to(self.model.device)
            cache = DynamicCache()
            self.model(input_ids=ids, past_key_values=cache, use_cache=True)
            self.prefix_ids, self.prefix_cache = ids, cache
        except Exception as e:
            print(f"⚠️  Sem cache de prefixo ({type(e).__name__}: {e}); o Jeffrey só fica mais lento.")

    @property
    def available(self) -> bool:
        return self.enabled and self.ready.is_set() and self.model is not None

    # -- montagem do contexto ------------------------------------------------------------
    def _project_facts(self) -> str:
        f, r = self.facts, self.rules
        tags = ", ".join(f"{t} ({n})" for t, n in r.tag_counts.most_common())
        linhas = [
            f"- Modelo de busca: {f.base_model} (SigLIP 2, multilíngue), ajustado em português; vetores de {f.dim} "
            "dimensões; imagens e textos no mesmo espaço.",
            f"- Catálogo: {fmt(len(r.records), 0)} imagens do Flickr8k com {fmt(sum(len(x['captions']) for x in r.records), 0)} "
            "legendas em português (cerca de 5 por imagem).",
            f"- Tags automáticas (contagem): {tags}. Com várias tags no filtro, a imagem precisa ter todas.",
            f"- Fine-tuning: partimos do SigLIP 2 já treinado pelo Google. O encoder de imagem ficou congelado; só as "
            f"últimas {f.text_layers} camadas do encoder de texto foram ajustadas, com pares (foto, legenda em português) "
            f"de {fmt(f.n_train, 0)} imagens de treino, {f.epochs} épocas, cerca de {fmt(f.train_minutes)} min em CPU, "
            "perda sigmoide. Objetivo: entender melhor frases em português. Não houve CLIP nem destilação nesse treino.",
            f"- Cada busca compara a consulta com TODAS as {fmt(len(r.records), 0)} imagens do catálogo (as de treino "
            "também aparecem nas buscas).",
            f"- Validação ({fmt(f.n_val, 0)} imagens fora do treino), texto→imagem: R@1 base {f.r('base', 't2i_R@1')}, "
            f"ajustado {f.r('tuned', 't2i_R@1')}; R@5 base {f.r('base', 't2i_R@5')}, ajustado {f.r('tuned', 't2i_R@5')}. "
            f"Imagem→texto R@1: base {f.r('base', 'i2t_R@1')}, ajustado {f.r('tuned', 'i2t_R@1')}. "
            "R@K = % das buscas em que a imagem certa aparece entre as K primeiras.",
            f"- O CLIP original da OpenAI, testado antes, fazia R@1 {fmt(f.clip_base_t2i_r1)}% em português "
            f"({fmt(f.clip_tuned_t2i_r1)}% ajustado).",
            "- IMPORTANTE: na busca texto→imagem, a frase é comparada com a FOTO (pixels), não com a legenda. A "
            "legenda é só para a pessoa ler; por isso o ranking pode não seguir as palavras das legendas, e uma foto "
            "pode ter coisas que a legenda não cita.",
            "- Score = similaridade de cosseno entre vetores normalizados. LIMITES TEÓRICOS: máximo 1 (vetores "
            "idênticos) e mínimo -1 (vetores opostos). Não é porcentagem de certeza: serve para ordenar.",
        ]
        s = self.score_stats
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
    def _search_facts(ctx: dict | None) -> str:
        if not ctx or not ctx.get("results"):
            return "Nenhuma busca feita nesta aba ainda."
        kind = ctx["kind"]
        nome = {"texto": "texto → imagem", "imagem": "imagem → imagem (a pessoa enviou uma foto)",
                "descrever": "imagem → texto (legendas do catálogo mais próximas da foto enviada)",
                "comparar": "comparação base × ajustado (lista abaixo = modelo ajustado)"}[kind]
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

    def _references(self, msg: str, k: int = 2) -> list[tuple[str, str]]:
        if self.embedder is None or self.kb_vecs is None or not len(self.kb_vecs):
            return []
        v = self.embedder.encode([msg])[0]
        sims = self.kb_vecs @ v
        idx = [i for i in np.argsort(-sims)[:k] if sims[i] >= 0.45]
        return [(self.rules.kb.questions[i], self.rules.kb.answers[i]) for i in idx]

    def build_messages(self, msg: str, ctx: dict | None, history: list[dict], kind: str) -> list[dict]:
        msgs = self._static_messages()
        for m in (history or [])[-2 * HISTORY_TURNS:]:
            content = m.get("content")
            if isinstance(content, list):  # formato do Gradio 6: [{"type": "text", "text": ...}]
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            if m.get("role") in ("user", "assistant") and isinstance(content, str) and content:
                msgs.append({"role": m["role"], "content": content})
        # tudo o que muda vai junto da pergunta: fica fora do cache e perto de onde o modelo presta mais atenção
        extra = ["[Última busca nesta aba]\n" + self._search_facts(ctx)]
        if (refs := self._references(msg)):
            extra.append("[Referências: perguntas parecidas já respondidas; use só se servirem]\n" +
                         "\n".join(f"P: {q}\nR: {a}" for q, a in refs))
        if (analise := self.rules.analysis(msg, ctx)):
            extra.append(f"[Dados do sistema para esta pergunta, corretos (use-os na resposta)]\n{analise}")
        extra.append("[Lembrete: só responda se for sobre o VisioVetor, imagens, buscas ou IA; se for outro assunto, "
                     "como geografia, esportes ou receitas, recuse com gentileza sem dar a resposta.]")
        msgs.append({"role": "user", "content": "\n\n".join(extra) + f"\n\nPergunta: {msg}"})
        return msgs

    # -- geração com streaming -------------------------------------------------------------------
    def stream(self, msg: str, ctx: dict | None, history: list[dict], kind: str) -> Iterator[str]:
        from transformers import TextIteratorStreamer
        msgs = self.build_messages(msg, ctx, history, kind)
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to(self.model.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        n = 0 if self.prefix_ids is None else self.prefix_ids.shape[1]
        if n and torch.equal(ids["input_ids"][0, :n], self.prefix_ids[0]):
            ids["past_key_values"] = copy.deepcopy(self.prefix_cache)  # reaproveita o começo fixo
        kwargs = dict(**ids, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                      repetition_penalty=1.05, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        with self._lock:  # um texto por vez: a CPU já usa todos os núcleos
            t = threading.Thread(target=lambda: self.model.generate(**kwargs), daemon=True)
            t.start()
            text = ""
            for piece in streamer:
                text += piece
                yield text.strip()
            t.join()


def score_stats(images, captions, sample: int = 300, seed: int = config.SEED) -> dict:
    """Faixas reais de score no índice (para o Jeffrey não chutar)."""
    rng = np.random.default_rng(seed)
    ci = rng.choice(len(captions.vectors), size=min(sample, len(captions.vectors)), replace=False)
    t2i = (captions.vectors[ci] @ images.vectors.T).max(1)  # melhor imagem para cada legenda
    ii = rng.choice(len(images.vectors), size=min(sample, len(images.vectors)), replace=False)
    sims = images.vectors[ii] @ images.vectors.T
    sims[np.arange(len(ii)), ii] = -1  # ignora a própria imagem
    full = captions.vectors[ci] @ images.vectors.T
    return {"t2i_p5": float(np.percentile(t2i, 5)), "t2i_med": float(np.median(t2i)),
            "t2i_p95": float(np.percentile(t2i, 95)), "t2i_max": float(full.max()),
            "all_min": float(full.min()), "all_med": float(np.median(full)),
            "i2i_med": float(np.median(sims.max(1)))}
