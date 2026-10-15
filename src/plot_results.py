"""Gera gráficos para a documentação a partir de results/ e outputs/.

Saídas: docs/img/experimentos_f1.png e docs/img/curvas_loss.png
Uso: python src/plot_results.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)


def plot_experiments():
    df = pd.read_csv(ROOT / "results" / "experiments.csv")
    base = json.loads((ROOT / "results" / "eval_base.json").read_text())
    names = ["base"] + df["name"].tolist()
    f1 = [base["token_f1"]] + df["token_f1"].tolist()
    st = [base["structure"]] + df["structure"].tolist()
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([i - 0.2 for i in x], f1, 0.4, label="token F1")
    ax.bar([i + 0.2 for i in x], st, 0.4, label="estrutura (4 seções)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30)
    ax.set_ylim(0, 1.05)
    ax.set_title("Modelo base vs. configurações de fine-tuning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMG / "experimentos_f1.png", dpi=150)


def plot_loss():
    fig, ax = plt.subplots(figsize=(8, 4))
    for d in sorted((ROOT / "outputs").glob("*/loss_history.json")):
        hist = json.loads(d.read_text())
        pts = [(h["step"], h["loss"]) for h in hist if "loss" in h]
        if pts:
            ax.plot(*zip(*pts), label=d.parent.name)
    ax.set_xlabel("passo")
    ax.set_ylabel("loss de treino")
    ax.set_title("Curvas de loss por experimento")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMG / "curvas_loss.png", dpi=150)


if __name__ == "__main__":
    plot_experiments()
    plot_loss()
    print("Gráficos salvos em", IMG)
