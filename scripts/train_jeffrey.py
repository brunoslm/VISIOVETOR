"""Ajusta um LLM pequeno para virar o Jeffrey (SFT com LoRA, opcionalmente DPO).

Usa os datasets gerados por scripts/generate_dataset.py:
  data/processed/jeffrey_sft_{train,val}.jsonl   (conversas no formato "messages")
  data/processed/jeffrey_dpo_{train,val}.jsonl   (prompt / chosen / rejected)

Dependências extras:  pip install trl peft

Uso:
  python scripts/train_jeffrey.py                          # SFT com Qwen2.5 0.5B Instruct
  python scripts/train_jeffrey.py --dpo                    # SFT e depois DPO
  python scripts/train_jeffrey.py --max-steps 5            # teste rápido do pipeline
  python scripts/train_jeffrey.py --model HuggingFaceTB/SmolLM2-135M-Instruct --max-steps 2

Em CPU, o SFT completo do modelo de 0,5B leva horas; com GPU, poucos minutos.
O resultado (adaptador LoRA) fica em models/jeffrey-lora/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.persona import SYSTEM_PROMPT  # noqa: E402

PERGUNTAS_TESTE = [
    "Oi, quem é você?",
    "Qual a precisão do modelo?",
    "Dá pra usar isso pra reconhecer rostos?",
    "Você é inútil",
]


def generate(model, tok, pergunta: str, max_new_tokens: int = 120) -> str:
    import torch

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": pergunta}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data-dir", default=str(config.PROCESSED_DIR))
    ap.add_argument("--output", default=str(config.MODELS_DIR / "jeffrey-lora"))
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=-1, help="limita os passos (útil para testar)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--dpo", action="store_true", help="depois do SFT, roda DPO com os pares de preferência")
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        sys.exit("❌ Instale as dependências extras: pip install trl peft")

    data = Path(args.data_dir)
    sft_files = {"train": str(data / "jeffrey_sft_train.jsonl"), "validation": str(data / "jeffrey_sft_val.jsonl")}
    if not Path(sft_files["train"]).exists():
        sys.exit("❌ Datasets não encontrados. Rode antes: python scripts/generate_dataset.py")
    ds = load_dataset("json", data_files=sft_files)
    print(f"📚 SFT: {len(ds['train'])} conversas de treino / {len(ds['validation'])} de validação | modelo={args.model}")

    peft_config = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                             target_modules="all-linear", task_type="CAUSAL_LM")
    out = Path(args.output)
    gpu = config.DEVICE == "cuda"
    common = dict(
        bf16=gpu, use_cpu=config.DEVICE == "cpu",
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, logging_steps=5, eval_strategy="epoch",
        save_strategy="no", report_to="none", max_steps=args.max_steps, seed=config.SEED,
    )
    sft = SFTTrainer(
        model=args.model,
        args=SFTConfig(output_dir=str(out / "sft"), num_train_epochs=args.epochs, learning_rate=args.lr,
                       max_length=args.max_length, **common),
        train_dataset=ds["train"], eval_dataset=ds["validation"], peft_config=peft_config,
    )
    sft.train()
    sft.save_model(str(out))
    tok = sft.processing_class
    model = sft.model

    if args.dpo:
        from trl import DPOConfig, DPOTrainer

        dpo_files = {"train": str(data / "jeffrey_dpo_train.jsonl"), "validation": str(data / "jeffrey_dpo_val.jsonl")}
        dds = load_dataset("json", data_files=dpo_files).remove_columns(["tipo_rejeicao"])
        print(f"⚖️  DPO: {len(dds['train'])} pares de treino / {len(dds['validation'])} de validação")
        model = model.merge_and_unload()  # o SFT vira a base; o DPO treina um LoRA novo por cima
        dpo = DPOTrainer(
            model=model,
            args=DPOConfig(output_dir=str(out / "dpo"), num_train_epochs=1, learning_rate=args.lr / 4,
                           beta=args.dpo_beta, max_length=args.max_length, **common),
            train_dataset=dds["train"], eval_dataset=dds["validation"], processing_class=tok,
            peft_config=peft_config,
        )
        dpo.train()
        dpo.save_model(str(out))
        model = dpo.model

    tok.save_pretrained(str(out))
    print(f"\n✅ Adaptador salvo em {out}\n\n🗣️  Respostas do modelo ajustado:")
    model.eval()
    for p in PERGUNTAS_TESTE:
        print(f"\n> {p}\n{generate(model, tok, p)}")


if __name__ == "__main__":
    main()
