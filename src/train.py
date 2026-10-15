"""Fine-tuning LoRA em GPU de um LLM para o assistente de manutenção industrial.

Exemplo:
  python src/train.py --name baseline --lora-r 8 --lr 2e-4 --epochs 3
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from common import DEFAULT_MODEL, SYSTEM_PROMPT, best_dtype, gpu_summary, load_jsonl, require_gpu

ROOT = Path(__file__).resolve().parent.parent


class SFTDataset(torch.utils.data.Dataset):
    """Tokeniza (prompt + resposta) e mascara o prompt: a perda é calculada só na resposta."""

    def __init__(self, rows, tokenizer, max_len):
        self.items = []
        for r in rows:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["instruction"]},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full = prompt + r["output"] + tokenizer.eos_token
            prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
            labels = [-100] * prompt_len + ids[prompt_len:]
            self.items.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels[: len(ids)]})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--name", default="baseline", help="nome do experimento (pasta em outputs/)")
    p.add_argument("--train", default=str(ROOT / "data" / "train.jsonl"))
    p.add_argument("--val", default=str(ROOT / "data" / "val.jsonl"))
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=None, help="padrão: 2 x r")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (economiza VRAM)")
    return p.parse_args()


def main():
    args = parse_args()
    require_gpu()
    set_seed(args.seed)
    out_dir = ROOT / "outputs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(gpu_summary())

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = best_dtype()
    use_bf16 = dtype == torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to("cuda")
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha or 2 * args.lora_r,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    for p in model.parameters():  # parâmetros treináveis em fp32 (necessário para AMP fp16)
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    train_ds = SFTDataset(load_jsonl(args.train), tokenizer, args.max_len)
    val_ds = SFTDataset(load_jsonl(args.val), tokenizer, args.max_len)
    print(f"treino: {len(train_ds)} | validação: {len(val_ds)}")

    targs = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    train_seconds = time.time() - t0
    eval_metrics = trainer.evaluate()

    model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")

    eval_loss = eval_metrics["eval_loss"]
    metrics = {
        "name": args.name,
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha or 2 * args.lora_r,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "train_loss": result.training_loss,
        "eval_loss": eval_loss,
        "perplexity": math.exp(eval_loss),
        "train_seconds": round(train_seconds, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "loss_history.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
