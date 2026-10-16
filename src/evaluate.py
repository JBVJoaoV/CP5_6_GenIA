"""Avalia o modelo (base ou com adaptador LoRA) no conjunto de validação, usando GPU.

Métricas:
  - token_f1: sobreposição de palavras entre a resposta gerada e a de referência
  - structure: fração das 4 seções obrigatórias presentes na resposta (formato correto)
  - seconds / tokens_per_s: latência de geração

Exemplos:
  python src/evaluate.py --tag base
  python src/evaluate.py --adapter outputs/baseline/adapter --tag baseline
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

from common import DEFAULT_MODEL, SECTIONS, Assistant, load_jsonl

ROOT = Path(__file__).resolve().parent.parent


def tokens(text):
    return re.findall(r"\w+", text.lower())


def token_f1(pred, ref):
    p, r = Counter(tokens(pred)), Counter(tokens(ref))
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / sum(p.values()), overlap / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def structure_score(text):
    return sum(s in text for s in SECTIONS) / len(SECTIONS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default=str(ROOT / "data" / "val.jsonl"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.adapter and not Path(args.adapter).exists():
        # sem isso, um treino que falhou faria a avaliação rodar silenciosamente o modelo BASE
        raise SystemExit(f"Adaptador não encontrado: {args.adapter}. O treino falhou? Veja o erro do train.py.")

    torch.manual_seed(args.seed)  # amostragem (temperatura > 0) reprodutível
    assistant = Assistant(args.model, args.adapter)
    rows = load_jsonl(args.data)
    records = []
    for r in rows:
        pred, stats = assistant.generate(
            r["instruction"],
            use_adapter=args.adapter is not None,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        records.append(
            {
                "instruction": r["instruction"],
                "reference": r["output"],
                "prediction": pred,
                "token_f1": token_f1(pred, r["output"]),
                "structure": structure_score(pred),
                "seconds": stats["seconds"],
                "tokens": stats["tokens"],
            }
        )

    df = pd.DataFrame(records)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / f"eval_{args.tag}.csv", index=False, encoding="utf-8")
    summary = {
        "tag": args.tag,
        "adapter": args.adapter,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "token_f1": round(df.token_f1.mean(), 4),
        "structure": round(df.structure.mean(), 4),
        "tokens_per_s": round(df.tokens.sum() / df.seconds.sum(), 1),
        "n": len(df),
    }
    (out / f"eval_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
