"""Gera data/train.jsonl e data/val.jsonl a partir de data/seed_examples.jsonl.

- Divide os exemplos em treino/validação (80/20, com semente fixa).
- No treino, cada relato é reescrito com prefixos diferentes (augmentation simples).
- A validação permanece com os relatos originais, para uma avaliação justa.

Uso: python src/prepare_data.py
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "data" / "seed_examples.jsonl"
PREFIXES = ["", "Equipamento com problema: ", "Operador relatou: ", "Chamado de manutenção - "]


def main(val_ratio=0.2, seed=42):
    rows = [json.loads(l) for l in SEED.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(seed).shuffle(rows)
    n_val = max(1, round(len(rows) * val_ratio))
    val, train = rows[:n_val], rows[n_val:]

    train_aug = [
        {"instruction": p + r["instruction"], "output": r["output"]}
        for r in train
        for p in PREFIXES
    ]
    random.Random(seed).shuffle(train_aug)

    for name, data in (("train", train_aug), ("val", val)):
        path = ROOT / "data" / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data) + "\n", encoding="utf-8")
        print(f"{name}: {len(data)} exemplos -> {path}")


if __name__ == "__main__":
    main()
