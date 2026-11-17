# Datasets da persona Jeffrey

> Gerado por `python scripts/generate_dataset.py`. Não edite à mão: mude o gerador e rode de novo.

O **Jeffrey** é o assistente do VisioVetor: cordial, educado e honesto. Ele só cita números que vêm do
sistema, admite quando não sabe e recusa com gentileza o que foge do escopo.

| Arquivo | Formato | Exemplos | Palavras por resposta | Conteúdo |
|---|---|---:|---:|---|
| `identidade.jsonl` | chat | 97 | 24 | nome, saudações, despedidas, tom e como lidar com grosseria (várias formas de perguntar) |
| `conhecimento_projeto.jsonl` | chat | 124 | 43 | perguntas e respostas sobre o VisioVetor: modelo, métricas, score, tags, uso |
| `limites.jsonl` | chat | 38 | 34 | recusas gentis: privacidade, saúde, métricas inventadas, pedidos fora do escopo |
| `comportamento.jsonl` | chat | 8 | 48 | casos de comportamento originais do projeto, na voz do Jeffrey |
| `treino_chat.jsonl` | chat | 19 | 30 | conversas com várias trocas escritas à mão (suporte, dúvidas, refinamento) |
| `suporte_tecnico.jsonl` | chat | 32 | 31 | erros comuns de instalação e uso, com a solução |
| `uso_do_app.jsonl` | chat | 50 | 41 | como usar o app e o chat do Jeffrey: o que perguntar, buscar pelo chat, treino, fotos próprias |
| `conceitos_ia.jsonl` | chat | 67 | 44 | conceitos de IA explicados pelo Jeffrey (embedding, cosseno, LoRA, DPO...) |
| `seguranca.jsonl` | chat | 15 | 33 | tentativas de manipulação, injeção de instruções e buscas preconceituosas |
| `explicacao_resultados.jsonl` | chat | 80 | 128 | explica buscas REAIS de texto→imagem (scores verdadeiros, às vezes com os passos) |
| `imagem_para_imagem.jsonl` | chat | 50 | 76 | explica buscas REAIS de imagens parecidas |
| `imagem_para_texto.jsonl` | chat | 50 | 80 | explica legendas recuperadas e zero-shot REAIS para uma foto |
| `refinamento_busca.jsonl` | chat | 54 | 70 | busca vaga → sugestões tiradas das legendas reais do catálogo |
| `catalogo.jsonl` | chat | 73 | 35 | contagens reais por tag e "tem foto de X?" respondido com honestidade |
| `comparacao_modelos.jsonl` | chat | 40 | 73 | compara resultados REAIS do modelo base e do ajustado |
| `conversas_geradas.jsonl` | chat | 40 | 48 | saudação → busca real explicada → refinamento → despedida |
| `treino_instrucao.jsonl` | instrução | 33 | 28 | formato instruction / input / output (estilo Alpaca) |
| `preferencias_dpo.jsonl` | preferência | 266 | 44 | prompt / chosen / rejected (TRL), à mão + rejeições geradas |
| `jeffrey_sft_train.jsonl` | chat | 783 | 52 | combinado (treino/validação) |
| `jeffrey_sft_val.jsonl` | chat | 87 | 52 | combinado (treino/validação) |
| `jeffrey_dpo_train.jsonl` | preferência | 239 | 44 | combinado (treino/validação) |
| `jeffrey_dpo_val.jsonl` | preferência | 27 | 48 | combinado (treino/validação) |

## Formatos

- **chat**: `{"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}`, compatível com o
  `SFTTrainer` do TRL, Axolotl e LLaMA-Factory.
- **instrução**: `{"system", "instruction", "input", "output"}` (estilo Alpaca).
- **preferência**: `{"prompt": [...], "chosen": [...], "rejected": [...]}`, compatível com o `DPOTrainer` do TRL.
  O campo extra `tipo_rejeicao` diz como a resposta rejeitada quebra a persona (certeza: 53, nome_errado: 41, grosseria: 41, ingles: 39, emoji: 38, seco: 27, manual: 16, numeros: 11).

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
