"""Frontend (Gradio) para o assistente de manutenção industrial.

Uso: python src/app.py [--adapter outputs/best/adapter]
Abre em http://127.0.0.1:7860
"""
import argparse
from pathlib import Path

import gradio as gr

from common import DEFAULT_MODEL, Assistant, gpu_summary

ROOT = Path(__file__).resolve().parent.parent
MODE_FT = "Fine-tuned (LoRA)"
MODE_BASE = "Modelo base"

EXAMPLES = [
    "O motor da esteira está aquecendo e o relé térmico desarma.",
    "A bomba de água faz barulho de cavitação e a vazão está baixa.",
    "O inversor de frequência mostra falha de sobretensão na desaceleração.",
    "O cilindro hidráulico da prensa está vazando óleo pela haste.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=str(ROOT / "outputs" / "best" / "adapter"))
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    assistant = Assistant(args.model, args.adapter)
    modes = [MODE_FT, MODE_BASE] if assistant.has_adapter else [MODE_BASE]

    def respond(prompt, mode, temperature, top_p, max_new_tokens):
        if not prompt.strip():
            return "Digite o relato do problema.", ""
        text, st = assistant.generate(
            prompt,
            use_adapter=(mode == MODE_FT),
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        info = f"{st['tokens']} tokens em {st['seconds']:.2f}s ({st['tokens'] / max(st['seconds'], 1e-6):.1f} tok/s)"
        return text, info

    with gr.Blocks(title="Assistente de Manutenção Industrial") as demo:
        gr.Markdown("# Assistente de Manutenção Industrial\nDescreva o problema do equipamento e receba causa provável, verificações, ação e segurança.")
        gr.Markdown(f"`{gpu_summary()}`")
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="Relato do problema", lines=4, placeholder="Ex.: A bomba centrífuga está vibrando e com ruído...")
                btn = gr.Button("Gerar resposta", variant="primary")
                gr.Examples(EXAMPLES, inputs=prompt)
            with gr.Column(scale=2):
                mode = gr.Radio(modes, value=modes[0], label="Modelo")
                temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="Temperatura (0 = determinístico)")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
                max_new = gr.Slider(64, 512, value=256, step=16, label="Máximo de tokens novos")
        answer = gr.Textbox(label="Resposta", lines=12)
        stats = gr.Markdown()
        btn.click(respond, [prompt, mode, temperature, top_p, max_new], [answer, stats])
        prompt.submit(respond, [prompt, mode, temperature, top_p, max_new], [answer, stats])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
