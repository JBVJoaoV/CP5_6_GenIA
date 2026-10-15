# Assistente de Manutenção Industrial com LLM local (fine-tuning LoRA em GPU)

**CP2/CP3 – Generative AI for Engineering**

> Grupo: _preencher nomes e RMs dos 4–5 integrantes_

Projeto que ajusta (fine-tuning com LoRA) um LLM open-source **rodando localmente em GPU** para atuar como assistente de diagnóstico de falhas em equipamentos industriais. O usuário descreve o problema em linguagem natural e o modelo responde em um formato padronizado de quatro seções: **Causa provável, Verificações, Ação recomendada e Segurança**.

## 1. Problema escolhido

Equipes de manutenção recebem relatos informais ("o motor está esquentando e desarmando") e precisam transformá-los rapidamente em um roteiro de diagnóstico seguro e padronizado. LLMs genéricos respondem de forma longa, variável e muitas vezes sem pensar em segurança. O objetivo do fine-tuning é fazer um modelo pequeno, rodando localmente, responder **sempre em português, no formato de 4 seções, com foco em causa raiz e segurança (LOTO, NR-10, NR-12, NR-13)**.

## 2. Modelo escolhido

| Opção | Parâmetros | VRAM aprox. (treino LoRA, bf16/fp16) | Quando usar |
|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0,5 B | ~3–4 GB | GPUs de 4–6 GB |
| **`Qwen/Qwen2.5-1.5B-Instruct`** (padrão) | 1,5 B | ~7–9 GB | GPUs de 8 GB ou mais |

Por que Qwen2.5-Instruct: licença aberta, bom desempenho em português para o tamanho, template de chat nativo e boa compatibilidade com Transformers/PEFT. Para trocar de modelo, use `--model` em qualquer script. Se faltar VRAM, use `--model Qwen/Qwen2.5-0.5B-Instruct`, `--batch-size 1` e/ou `--grad-ckpt`.

**Técnica:** LoRA (Low-Rank Adaptation) via biblioteca PEFT nas camadas de atenção e MLP. Só uma pequena fração dos parâmetros é treinada, o que cabe em GPUs de consumo.

## 3. Dataset

- `data/seed_examples.jsonl`: 59 pares (relato → resposta estruturada) cobrindo motores, bombas, compressores, correias, redutores, inversores, CLPs, hidráulica, pneumática, soldagem, caldeira, transformador, CNC etc.
- `src/prepare_data.py` divide em treino/validação (80/20, semente 42) e, **somente no treino**, reescreve cada relato com 4 prefixos diferentes (augmentation simples). A validação mantém os relatos originais.
- Split resultante: **47 relatos de treino (188 com augmentation) e 12 de validação**.
- **Limitação:** o conjunto é pequeno (12 exemplos de validação dão uma estimativa ruidosa). Para melhorar os resultados, o grupo pode ampliar o `seed_examples.jsonl` (meta: 100+ exemplos) mantendo o mesmo formato.

> Revisão técnica: as respostas do dataset são material didático e devem ser revisadas por um especialista antes de qualquer uso real em manutenção.

## 4. Estrutura do repositório

```
├── data/seed_examples.jsonl     # exemplos base
├── src/
│   ├── check_gpu.py             # informações da GPU + benchmark (evidência)
│   ├── prepare_data.py          # gera train/val
│   ├── common.py                # prompt, dados, carregamento e geração
│   ├── train.py                 # fine-tuning LoRA em GPU
│   ├── evaluate.py              # avaliação (F1, estrutura, velocidade)
│   ├── run_experiments.py       # teste de diferentes parâmetros
│   ├── plot_results.py          # gráficos para o relatório
│   └── app.py                   # frontend Gradio
├── results/                     # CSVs/JSONs das avaliações e experimentos
├── docs/img/                    # prints da GPU e gráficos
├── requirements.txt
└── README.md
```

## 5. Instalação (GPU + CUDA)

1. **Descubra a GPU e o driver** no terminal do VSCode: `nvidia-smi` (mostra o modelo da placa e a versão máxima de CUDA suportada pelo driver). Atualize o driver NVIDIA se estiver antigo.
2. **Escolha a versão de CUDA correta para a sua placa** no seletor oficial do PyTorch (https://pytorch.org/get-started/locally/). Como regra geral: placas RTX 50xx (Blackwell) exigem builds CUDA 12.8 ou superior; RTX 20/30/40 funcionam com CUDA 12.x. Os wheels do PyTorch já incluem as bibliotecas CUDA necessárias; o CUDA Toolkit completo só é preciso se o grupo quiser compilar extensões ou usar `nvcc`.
3. Crie o ambiente e instale (exemplo com CUDA 12.4; troque `cu124` pela versão que o seletor indicar):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python src/check_gpu.py      # deve mostrar "CUDA disponível: True" e o nome da GPU
```

Para o professor ver o CUDA instalado, rode também `nvcc --version` (só existe se o CUDA Toolkit estiver instalado; para o treino em si basta o driver + wheel do PyTorch com CUDA).

Observação: o primeiro uso baixa o modelo do Hugging Face (~1–3 GB).

## 6. Como executar

```bash
python src/prepare_data.py                 # gera data/train.jsonl e data/val.jsonl
python src/train.py --name baseline        # um treino (LoRA r=8, lr=2e-4, 3 épocas)
python src/evaluate.py --tag base          # avalia o modelo SEM fine-tuning
python src/evaluate.py --adapter outputs/baseline/adapter --tag baseline
python src/run_experiments.py              # bateria completa (o tempo varia conforme a GPU; anote o tempo real no relatório) de experimentos (use --quick para uma versão rápida)
python src/plot_results.py                 # gráficos em docs/img/
python src/app.py                          # frontend em http://127.0.0.1:7860
```

Todos os scripts abortam com erro se não houver GPU CUDA, garantindo que os testes foram feitos em GPU.

## 7. Parâmetros testados

`run_experiments.py` varia **um parâmetro por vez** a partir da configuração baseline e depois varre a temperatura de geração na melhor configuração.

| Experimento | LoRA r | Learning rate | Épocas |
|---|---|---|---|
| baseline | 8 | 2e-4 | 3 |
| r4 | 4 | 2e-4 | 3 |
| r16 | 16 | 2e-4 | 3 |
| lr1e-4 | 8 | 1e-4 | 3 |
| lr5e-4 | 8 | 5e-4 | 3 |
| ep5 | 8 | 2e-4 | 5 |
| ep8 | 8 | 2e-4 | 8 |

Geração: temperatura em {0.0, 0.3, 0.7, 1.0} com top_p = 0.9 (semente fixa 42 para a amostragem).

## 8. Avaliação

Conjunto de validação (relatos originais, nunca usados no treino). Métricas:

- **eval_loss / perplexidade** no treino (`outputs/<exp>/metrics.json`);
- **token F1**: sobreposição de palavras entre a resposta gerada e a de referência;
- **estrutura**: fração das 4 seções obrigatórias presentes (mede se o modelo aprendeu o formato);
- **tokens/s** e **VRAM de pico** (desempenho em GPU);
- **avaliação qualitativa**: leitura manual das respostas em `results/eval_<tag>.csv`, comparando base vs. fine-tuned (precisão técnica, coerência e segurança).

### Resultados

_Preencher após rodar `python src/run_experiments.py` na máquina com GPU (copie de `results/experiments.csv` e `results/gen_sweep.csv`)._

| Experimento | token F1 | estrutura | eval_loss | perplexidade | tempo de treino (s) | VRAM pico (GB) |
|---|---|---|---|---|---|---|
| base (sem fine-tuning) | | | – | – | – | – |
| baseline | | | | | | |
| ... | | | | | | |

![Comparação](docs/img/experimentos_f1.png)
![Loss](docs/img/curvas_loss.png)

### Análise (preencher)

- Qual configuração foi a melhor e por quê?
- O que mudou entre o modelo base e o fine-tuned (formato, idioma, segurança)?
- Houve sinais de overfitting (muitas épocas / learning rate alto)?
- Exemplos de acertos e erros (cole 2–3 respostas de `results/eval_*.csv`).
- Limitações e melhorias futuras (mais dados, avaliação por especialista, RAG com manuais).

## 9. Frontend

`python src/app.py` abre uma interface Gradio onde o usuário escreve o prompt (relato do problema), escolhe entre **modelo base** e **fine-tuned**, ajusta temperatura, top_p e tamanho máximo da resposta e vê a latência e a GPU em uso.

_Inserir print do frontend aqui: `docs/img/frontend.png`._

## 10. Evidência da GPU no VSCode

1. Instale as extensões **GPU Environment** e **GPU Monitor** no VSCode.
2. Abra o terminal integrado e rode `python src/check_gpu.py` (ou `nvidia-smi`) durante um treino.
3. Tire um print com o VSCode mostrando o painel da extensão e o terminal com a GPU em uso e salve em `docs/img/gpu_vscode.png`.

![GPU no VSCode](docs/img/gpu_vscode.png)

## 11. Checklist da entrega

- [ ] LLM escolhido e justificado (seção 2)
- [ ] Problema específico definido (seção 1)
- [ ] Diferentes parâmetros testados (seção 7)
- [ ] Modelo avaliado, respostas analisadas (seção 8)
- [ ] Testes feitos em GPU (print em `docs/img/gpu_vscode.png`)
- [ ] Frontend funcionando (print em `docs/img/frontend.png`)
- [ ] Código e documentação no GitHub (link enviado na plataforma)
- [ ] Nomes do grupo no topo do README

## 12. Publicando no GitHub

```bash
git init
git add .
git commit -m "CP2/CP3: fine-tuning LoRA para assistente de manutenção industrial"
git branch -M main
git remote add origin https://github.com/<usuario>/<repositorio>.git
git push -u origin main
```

Antes do `git add`, confirme que `docs/img/` tem os prints e que `results/` tem os CSVs/JSONs gerados na GPU (o `.gitignore` já exclui `outputs/`, `.venv/` e `data/train.jsonl`/`val.jsonl`, que são regenerados por `prepare_data.py`). Cada integrante deve ter commits no repositório.

## 13. Tecnologias

Python, PyTorch (CUDA), Hugging Face Transformers, PEFT (LoRA), Gradio, pandas, matplotlib.
