"""Mostra as informações da GPU e roda um teste rápido de desempenho.

Use este script (ou o nvidia-smi) no terminal do VSCode para tirar o print de evidência.
Uso: python src/check_gpu.py
"""
import subprocess
import time

import torch


def main():
    print("=" * 60)
    print("PyTorch:", torch.__version__)
    print("CUDA (compilado no PyTorch):", torch.version.cuda)
    print("CUDA disponível:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("\n[ERRO] Nenhuma GPU CUDA encontrada. Verifique o driver NVIDIA e instale o PyTorch com a versão de CUDA correta.")
        return

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    print("GPU:", props.name)
    print(f"VRAM total: {props.total_memory / 1024**3:.1f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")
    print("bf16 suportado:", torch.cuda.is_bf16_supported())
    print("=" * 60)

    # Benchmark simples: multiplicação de matrizes em GPU vs CPU
    n = 4096
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    t0 = time.time()
    a @ b
    cpu_t = time.time() - t0

    ag, bg = a.cuda().half(), b.cuda().half()
    torch.cuda.synchronize()
    ag @ bg  # aquecimento
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        ag @ bg
    torch.cuda.synchronize()
    gpu_t = (time.time() - t0) / 10
    print(f"matmul {n}x{n}: CPU {cpu_t*1000:.0f} ms | GPU (fp16) {gpu_t*1000:.1f} ms | speedup ~{cpu_t/gpu_t:.0f}x")
    print("=" * 60)

    try:
        print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    except FileNotFoundError:
        print("nvidia-smi não encontrado no PATH.")


if __name__ == "__main__":
    main()
