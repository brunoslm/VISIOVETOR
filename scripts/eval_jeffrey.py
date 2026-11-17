"""Avalia se o Jeffrey (modo regras) responde certo a perguntas reais de usuários.

Cada caso é (pergunta, trecho que a resposta certa precisa conter). A métrica é a taxa de
acerto: a pergunta precisa cair na resposta certa, não basta responder alguma coisa. Muitas
perguntas aqui são escritas de um jeito diferente dos datasets, para medir generalização.

Uso:
  python scripts/generate_dataset.py   # depois de mudar os datasets
  python scripts/eval_jeffrey.py       # -v mostra também os acertos
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import carregar_registros  # noqa: E402
from src.jeffrey_chat import JeffreyChat, norm  # noqa: E402
from src.persona import carregar_fatos  # noqa: E402

CASOS = [
    # uso
    ("me da umas dicas pra buscar", "quem, fazendo o quê"),
    ("como faço pra testar sem foto", "fotos de exemplo"),
    ("por que a busca não achou", "nenhuma busca"),  # sem busca feita, não há o que analisar
    ("por que errou a busca?", "motivos mais comuns"),
    ("tu enxerga as fotos?", "Não vejo as fotos"),
    ("como eu uso isso?", "Texto → Imagem"),
    ("posso buscar em inglês", "multilíngue"),
    # projeto e métricas
    ("o treino ajudou mesmo?", "dos erros"),
    ("por que o ganho foi pequeno", "dos erros que sobravam"),
    ("qual a taxa de acerto do sistema", "em 1º lugar"),
    ("quantas épocas", "épocas"),
    ("quantas dimensões tem cada vetor", "768"),
    ("qual a diferença entre r@1 e r@5", "entre as 5"),
    ("o que é o dataset flickr", "Flickr8k"),
    ("dá pra usar numa empresa?", "comercial"),
    ("funciona com placa nvidia?", "CUDA"),
    ("precisa de net pra rodar?", "primeira vez"),
    ("minhas fotos ficam salvas?", "salv"),
    ("é seguro mandar minha foto?", "localmente"),
    ("como escalar para milhões de imagens", "HNSW"),
    ("quanto tempo demorou pra treinar", "minutos"),
    # conceitos
    ("o que é um espaço latente", "ponto"),
    ("explica o rag", "buscadas na hora"),
    ("busca um cachorro na praia", "caixa"),  # o chat não busca: indica a caixa do topo
    ("o que é negativo difícil", "parecidos com o certo"),
    ("o modelo esquece o que sabia?", "congelamos"),
    ("o que é o produto escalar", "número a número"),
    ("o que é logit bias", "sigmoide"),
    ("me explica o siglip 2", "Google"),
    ("o que é zero-shot", "rótulos"),
    ("por que o score é tão baixo assim", "normal"),
    ("0,2 quer dizer 20%?", "Não é porcentagem"),
    # identidade
    ("vc usa gpt?", "Qwen"),
    ("quantos anos tu tem", "idade"),
    ("você é consciente?", "consciência"),
    ("tu é burro", "Sinto muito"),
    # fora do escopo
    ("quem é o presidente do brasil", "só consigo ajudar com o VisioVetor"),
    ("faz um código em javascript", "assistente do VisioVetor"),
    ("me dá uma receita de bolo", "Cozinha"),
    # catálogo e regras
    ("quantas fotos de cachorro tem", "\"cachorro\""),
    ("a primeira busca demorou muito", "carrega"),
]


def main() -> None:
    detalhado = "-v" in sys.argv
    jeff = JeffreyChat(carregar_registros(), fatos=carregar_fatos())
    erros = 0
    for pergunta, esperado in CASOS:
        resposta, _ = jeff.responder(pergunta, None)
        ok = norm(esperado) in norm(resposta)
        erros += not ok
        if detalhado or not ok:
            print(f"{'ok  ' if ok else 'ERRO'} {pergunta}\n     → {resposta[:140]}")
    n = len(CASOS)
    print(f"\nJeffrey (regras): {n - erros}/{n} respostas certas ({100 * (n - erros) / n:.0f}%)")
    sys.exit(1 if erros else 0)


if __name__ == "__main__":
    main()
