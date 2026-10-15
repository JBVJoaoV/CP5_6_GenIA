"""Roda a bateria de experimentos (hiperparâmetros de treino + parâmetros de geração).

Passos:
  1. avalia o modelo base (sem fine-tuning);
  2. treina e avalia cada configuração (variando uma coisa por vez a partir da baseline);
  3. escolhe a melhor (maior token_f1) e copia para outputs/best/adapter;
  4. varre a temperatura de geração na melhor configuração.

Resultados: results/experiments.csv e results/gen_sweep.csv
Uso: python src/run_experiments.py [--quick] [--model Qwen/Qwen2.5-0.5B-Instruct]
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

BASELINE = {"lora_r": 8, "lr": 2e-4, "epochs": 3}
CONFIGS = {
    "baseline": {},
    "r4": {"lora_r": 4},
    "r16": {"lora_r": 16},
    "lr1e-4": {"lr": 1e-4},
    "lr5e-4": {"lr": 5e-4},
    "ep5": {"epochs": 5},
    "ep8": {"epochs": 8},
}
QUICK = ["baseline", "r16", "lr5e-4"]
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]


def run(cmd):
    print("\n>>>", " ".join(map(str, cmd)))
    subprocess.run([sys.executable, *map(str, cmd)], check=True, cwd=ROOT)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    model_args = ["--model", args.model] if args.model else []

    run([SRC / "prepare_data.py"])
    run([SRC / "evaluate.py", "--tag", "base", *model_args])

    names = QUICK if args.quick else list(CONFIGS)
    rows = []
    for name in names:
        cfg = {**BASELINE, **CONFIGS[name]}
        run([SRC / "train.py", "--name", name, "--lora-r", cfg["lora_r"], "--lr", cfg["lr"], "--epochs", cfg["epochs"], *model_args])
        run([SRC / "evaluate.py", "--adapter", f"outputs/{name}/adapter", "--tag", name, *model_args])
        m, e = read(ROOT / "outputs" / name / "metrics.json"), read(ROOT / "results" / f"eval_{name}.json")
        rows.append({**{k: m[k] for k in ("name", "lora_r", "lr", "epochs", "train_loss", "eval_loss", "perplexity", "train_seconds", "peak_vram_gb")},
                     "token_f1": e["token_f1"], "structure": e["structure"], "tokens_per_s": e["tokens_per_s"]})

    df = pd.DataFrame(rows).sort_values("token_f1", ascending=False)
    df.to_csv(ROOT / "results" / "experiments.csv", index=False)
    print("\n", df.to_string(index=False))

    best = df.iloc[0]["name"]
    print(f"\nMelhor configuração: {best}")
    best_dir = ROOT / "outputs" / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(ROOT / "outputs" / best, best_dir)

    sweep = []
    for t in TEMPERATURES:
        tag = f"best_T{t}"
        run([SRC / "evaluate.py", "--adapter", f"outputs/{best}/adapter", "--tag", tag, "--temperature", t, *model_args])
        e = read(ROOT / "results" / f"eval_{tag}.json")
        sweep.append({"temperature": t, "token_f1": e["token_f1"], "structure": e["structure"], "tokens_per_s": e["tokens_per_s"]})
    pd.DataFrame(sweep).to_csv(ROOT / "results" / "gen_sweep.csv", index=False)
    print(pd.DataFrame(sweep).to_string(index=False))


if __name__ == "__main__":
    main()
