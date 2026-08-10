<div align="center">
  <img src="ui/static/logo.svg" width="96" alt="onegpu-llm logo">
  <h1>onegpu-llm</h1>
  <p>Run open-source LLMs on a single consumer GPU — with a web console to manage everything.</p>
  <p>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/badge/engine-vLLM-green.svg" alt="Engine: vLLM">
    <img src="https://img.shields.io/badge/gpu-24GB%20VRAM-orange.svg" alt="GPU: 24GB VRAM">
  </p>
  <p><b>English</b> · <a href="README.zh-CN.md">中文</a> · <a href="README.es.md">Español</a></p>
</div>

---

## What is this?

A batteries-included workspace for deploying open-source LLMs (Qwen, Gemma, …) on a **single 24GB GPU** (tested on RTX 4090, WSL2 + Docker Desktop). Inference is served by **vLLM** with an OpenAI-compatible API, and a lightweight **web console** handles model download, start/stop switching, logs, and GPU monitoring.

Designed for personal AI toolchains: plug the local endpoint into [opencode](https://opencode.ai), Cline, Continue, or any OpenAI-compatible client.

## Features

- **Web console** (`http://localhost:9000`): one-click model download with live progress, start/stop, container logs, real-time GPU stats (VRAM / utilization / temperature / power), built-in chat panel and smoke test (chat + tool calling)
- **Chat panel**: multi-turn chat with model picker (defaults to the running model), Markdown/code rendering, image & text-file attachments (multimodal models supported), optional **web search** (free DuckDuckGo, no API key — results injected as grounded context to reduce hallucination, with source citations)
- **One model at a time** by design — switching models is one click, the console stops the running one automatically (24GB VRAM can't hold two)
- **Quantized-first model registry**: AWQ models tuned to fit 24GB, with per-model `max-model-len` budgets
- **Tool calling ready**: Hermes parser + auto tool choice enabled out of the box, works with agentic coding tools
- **Fast restarts**: vLLM compile cache is persisted to the workspace, warm restarts take ~70s
- **Everything stays in the workspace**: weights, caches, logs — nothing leaks into `$HOME` or other drives

## Quick start

Prerequisites: NVIDIA GPU (≥24GB recommended), Docker with GPU support (Docker Desktop + WSL2 works), Python 3.10+.

```bash
git clone https://github.com/ezhouyang/onegpu-llm.git
cd onegpu-llm

./scripts/download.sh qwen3-14b       # download weights via ModelScope (fast in CN)
docker compose --profile qwen3-14b up -d
./scripts/test.sh                     # smoke test: chat + tool calling
```

Then start the console:

```bash
./scripts/ui.sh        # http://localhost:9000
```

The OpenAI-compatible API listens on `http://localhost:8000/v1`.

## Model registry

| Profile | Model | Use case | VRAM (weights) | max-model-len |
|---|---|---|---|---|
| `qwen3-14b` | Qwen/Qwen3-14B-AWQ | daily driver, Chinese chat/RAG | ~9GB | 32768 |
| `qwen3-32b` | Qwen/Qwen3-32B-AWQ | best single-GPU quality (slow, tight context) | ~19GB | 8192 |
| `qwen3-30b-a3b` | Qwen/Qwen3-30B-A3B-Instruct-2507-AWQ | MoE, fast + near-32B quality | ~17GB | 16384 |
| `qwen25-coder-32b` | Qwen/Qwen2.5-Coder-32B-Instruct-AWQ | coding agent backend | ~19GB | 8192 |
| `gemma3-4b` | LLM-Research/gemma-3-4b-it | multimodal (image input), lightweight | ~8GB | 16384 |

Adding a model = one line in `scripts/download.sh` + one service block in `docker-compose.yml`. Gemma 3 27B (needs a community quant) and other open models planned.

## Why vLLM instead of Ollama?

| | onegpu-llm (vLLM) | Ollama |
|---|---|---|
| Throughput | PagedAttention + continuous batching, much higher under concurrency | serial per-request, batching limited |
| Tool calling | first-class (`--tool-call-parser`), stable for agentic tools | varies by model/template |
| Quantization | AWQ/GPTQ/FP8 — higher quality than GGUF at same VRAM | GGUF only |
| Fine control | every engine flag exposed via compose (KV cache dtype, max len, mm limits…) | limited knobs |
| Ease of use | needs Docker + some config (this repo handles it) | one-liner install, great for casual use |

For a single-user coding-agent backend on one GPU, vLLM gives better quality-per-VRAM (AWQ) and more reliable tool calling; Ollama is simpler for quick experiments.

## opencode integration

Add a provider pointing at the local endpoint in `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "onegpu": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:8000/v1" },
      "models": {
        "qwen25-coder-32b": { "name": "Qwen2.5 Coder 32B (local)" },
        "qwen3-14b": { "name": "Qwen3 14B (local)" }
      }
    }
  },
  "model": "onegpu/qwen25-coder-32b"
}
```

Then run `opencode` and chat as usual — requests go to your GPU. Notes:

- The served model name is set by `--served-model-name` in `docker-compose.yml`; keep it in sync with the `models` keys above
- Start the matching profile first (`gemma3-4b` is multimodal — handy for screenshot-driven debugging)
- For coding, `qwen25-coder-32b` is the recommended profile; tool calling is preconfigured
- Same endpoint works with Cline / Continue / LibreChat: OpenAI-compatible base URL `http://localhost:8000/v1`, any API key string

## Project layout

```
onegpu-llm/
├── docker-compose.yml     # model services, one profile per model
├── models/                # weights (git-ignored)
├── scripts/               # download.sh / test.sh / ui.sh
├── ui/                    # web console (FastAPI + single-page, no build step)
├── .cache/                # modelscope/pip caches (git-ignored)
└── logs/                  # server & download logs (git-ignored)
```

## Notes

- WSL2 users: vLLM ≥0.25 V2 model runner requires UVA which WSL2 lacks; this repo sets `VLLM_USE_V2_MODEL_RUNNER=0` already
- Qwen3 thinking mode: append `/no_think` to prompts if you want direct answers
- See `CLAUDE.md` for the full operating conventions (VRAM budgets, adding models, red lines)

## License

[MIT](LICENSE)

---

## Appendix: environment setup from scratch

Tested setup: Windows 11 + WSL2 (Ubuntu 22.04) + Docker Desktop, RTX 4090 24GB. Native Linux works too — skip the WSL parts.

### 1. NVIDIA driver

Install the latest **Windows** NVIDIA Game Ready / Studio driver. Nothing CUDA-related needs to be installed inside WSL — the driver is shared. Verify in WSL:

```bash
nvidia-smi    # should list your GPU
```

### 2. WSL2

```powershell
# in Windows PowerShell (admin)
wsl --install -d Ubuntu-22.04
```

Already on WSL1? Convert with `wsl --set-version Ubuntu-22.04 2`.

### 3. Docker Desktop

1. Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/), keep the default **WSL2 backend**
2. Settings → **Resources → WSL Integration** → enable your distro (e.g. Ubuntu-22.04)
3. Optional but recommended — move Docker's data disk off the C: drive (the vLLM image alone is ~28GB): Settings → Resources → **Advanced → Disk image location**, pick a folder on another drive

Verify in WSL:

```bash
docker version
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi   # GPU visible in container
```

### 4. Python

The scripts and web console need Python ≥3.10 with `pip` (Ubuntu 22.04's `python3` works). First download auto-installs the `modelscope` CLI.

### Disk layout

Everything this project writes stays inside the workspace (weights, caches, logs) — the only large external consumer is Docker's image store, handled in step 3.3.

---

<div align="center">
  中文文档请见 <a href="README.zh-CN.md">README.zh-CN.md</a> · Documentación en español: <a href="README.es.md">README.es.md</a>
</div>
