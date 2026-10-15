"""Funções compartilhadas: prompt de sistema, dados e carregamento/geração do modelo."""
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "Você é um assistente de manutenção industrial. Dado o relato de um problema em um "
    "equipamento, responda SEMPRE em português com quatro seções, nesta ordem: "
    "'Causa provável:', 'Verificações:', 'Ação recomendada:' e 'Segurança:'."
)
SECTIONS = ["Causa provável:", "Verificações:", "Ação recomendada:", "Segurança:"]


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def require_gpu():
    if not torch.cuda.is_available():
        raise SystemExit(
            "GPU CUDA não encontrada. Os testes deste trabalho devem rodar em GPU: "
            "verifique o driver NVIDIA e a instalação do PyTorch com CUDA (python src/check_gpu.py)."
        )
    return torch.device("cuda")


def gpu_summary():
    if not torch.cuda.is_available():
        return "GPU: não disponível"
    p = torch.cuda.get_device_properties(0)
    return f"GPU: {p.name} | VRAM: {p.total_memory / 1024**3:.1f} GB | CUDA {torch.version.cuda} | PyTorch {torch.__version__}"


def best_dtype():
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def build_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class Assistant:
    """Carrega o modelo base (e opcionalmente o adaptador LoRA) na GPU."""

    def __init__(self, model_name=DEFAULT_MODEL, adapter=None):
        self.device = require_gpu()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=best_dtype()).to(self.device)
        self.has_adapter = adapter is not None and Path(adapter).exists()
        if self.has_adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.eval()

    @torch.no_grad()
    def generate(self, question, use_adapter=True, temperature=0.0, top_p=0.9, max_new_tokens=256):
        prompt = build_prompt(self.tokenizer, question)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False
        ).to(self.device)
        gen_kwargs = dict(max_new_tokens=int(max_new_tokens), pad_token_id=self.tokenizer.pad_token_id)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=float(top_p))
        else:
            gen_kwargs.update(do_sample=False)

        ctx = self.model.disable_adapter() if (self.has_adapter and not use_adapter) else nullcontext()
        torch.cuda.synchronize()
        t0 = time.time()
        with ctx:
            out = self.model.generate(**inputs, **gen_kwargs)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text, {"seconds": elapsed, "tokens": int(new_tokens.shape[0])}
