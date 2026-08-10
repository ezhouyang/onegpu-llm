<div align="center">
  <img src="ui/static/logo.svg" width="96" alt="onegpu-llm logo">
  <h1>onegpu-llm</h1>
  <p>在单张消费级显卡上运行开源大模型 —— 配可视化 Web 管理台。</p>
  <p>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/badge/engine-vLLM-green.svg" alt="Engine: vLLM">
    <img src="https://img.shields.io/badge/gpu-24GB%20VRAM-orange.svg" alt="GPU: 24GB VRAM">
  </p>
  <p><a href="README.md">English</a> · <b>中文</b> · <a href="README.es.md">Español</a></p>
</div>

---

## 项目简介

面向**单卡 24GB 显存**（RTX 4090 实测，WSL2 + Docker Desktop）的开源大模型一站式部署工作空间。推理引擎采用 **vLLM**，对外提供 OpenAI 兼容 API；自带轻量 **Web 管理台**，覆盖模型下载、启停切换、日志查看、GPU 监控与对话测试全流程，无需敲命令行。

设计目标：给个人 AI 工具链（opencode、Cline、Continue 等）提供一个稳定、可管理的本地模型后端。

## 功能特性

- **Web 管理台**（`http://localhost:9000`）
  - 模型卡片：一键下载（实时进度日志）、一键启停（自动停掉旧模型，24G 显存单模型约束）、容器日志、GPU 实时监控（显存/利用率/温度/功耗）、冒烟测试（对话 + 工具调用）
  - **对话面板**：多轮对话；模型选择器（默认当前运行模型）；Markdown/代码高亮渲染；图片与文本文件附件（支持多模态模型）；**联网搜索开关**（免费 DuckDuckGo，无需 API Key，搜索结果注入上下文以减少幻觉，并附来源引用）
- **量化优先的模型清单**：AWQ 量化模型适配 24GB 显存，每个模型预设 `max-model-len` 预算
- **工具调用开箱即用**：Hermes parser + auto tool choice，可直接接 agentic coding 工具
- **快速重启**：vLLM 编译缓存持久化到工作空间，热重启约 70 秒
- **数据不出工作空间**：权重、缓存、日志全部在项目目录内，不污染 `$HOME` 和其他盘

## 快速开始

前置条件：NVIDIA 显卡（建议 ≥24GB 显存）、支持 GPU 的 Docker（Docker Desktop + WSL2 即可）、Python 3.10+。详细环境搭建见文末[附录](#附录从零搭建环境)。

```bash
git clone https://github.com/ezhouyang/onegpu-llm.git
cd onegpu-llm

./scripts/download.sh qwen3-14b          # 下载模型权重（ModelScope，国内速度快）
docker compose --profile qwen3-14b up -d # 启动推理服务
./scripts/test.sh                        # 冒烟测试：对话 + 工具调用
./scripts/ui.sh                          # 启动管理台 http://localhost:9000
```

OpenAI 兼容 API 地址：`http://localhost:8000/v1`

## 模型清单

| Profile | 模型 | 用途 | 权重显存 | max-model-len |
|---|---|---|---|---|
| `qwen3-14b` | Qwen/Qwen3-14B-AWQ | 主力，中文对话/RAG | ~9GB | 32768 |
| `qwen3-32b` | Qwen/Qwen3-32B-AWQ | 单机质量上限（慢，上下文紧） | ~19GB | 8192 |
| `qwen3-30b-a3b` | Qwen/Qwen3-30B-A3B-Instruct-2507-AWQ | MoE，速度快质量接近 32B | ~17GB | 16384 |
| `qwen25-coder-32b` | Qwen/Qwen2.5-Coder-32B-Instruct-AWQ | coding agent 后端 | ~19GB | 8192 |
| `gemma3-4b` | LLM-Research/gemma-3-4b-it | 多模态（图片输入），轻量 | ~8GB | 16384 |

**新增模型三步走**：`scripts/download.sh` 模型表加一行 → `docker-compose.yml` 复制一个 service 改路径和 profile 名 → 管理台自动出现新卡片。Gemma 3 27B（待社区量化版）等更多模型规划中。

## 为什么用 vLLM 而不是 Ollama？

| | onegpu-llm (vLLM) | Ollama |
|---|---|---|
| 吞吐 | PagedAttention + 连续批处理，并发下高得多 | 请求串行，批处理能力弱 |
| 工具调用 | 原生 `--tool-call-parser`，agentic 场景稳定 | 取决于模型/模板，不稳定 |
| 量化 | AWQ/GPTQ/FP8，同显存下质量优于 GGUF | 仅 GGUF |
| 可控性 | 引擎参数全部可通过 compose 调整（KV cache、上下文、多模态限制…） | 可调项少 |
| 上手 | 需要 Docker + 少量配置（本仓库已备好） | 一行安装，适合快速体验 |

单卡单用户的 coding agent 后端场景，vLLM 的「每显存质量」（AWQ）和工具调用可靠性更好；Ollama 更适合随手试模型。

## 接入 opencode

在 `~/.config/opencode/opencode.json` 添加 provider：

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

注意事项：

- 模型名以 `docker-compose.yml` 里 `--served-model-name` 为准，需与上面 `models` 的 key 一致
- 先启动对应 profile 再使用；coding 场景推荐 `qwen25-coder-32b`
- Cline / Continue / LibreChat 等工具同样可用：填 OpenAI 兼容地址 `http://localhost:8000/v1`，API Key 任意字符串

## 目录结构

```
onegpu-llm/
├── docker-compose.yml     # 模型服务定义，一个 profile 一个模型
├── models/                # 模型权重（不进 git）
├── scripts/               # download.sh / test.sh / ui.sh
├── ui/                    # Web 管理台（FastAPI + 单页面，无构建步骤）
├── .cache/                # modelscope / pip / vLLM 编译缓存（不进 git）
└── logs/                  # 服务与下载日志（不进 git）
```

## 使用提示

- WSL2 用户：vLLM ≥0.25 的 V2 runner 需要 UVA，WSL2 不支持，本仓库已设置 `VLLM_USE_V2_MODEL_RUNNER=0`
- Qwen3 思考模式：想要直接回答，在问题后加 `/no_think`
- 显存预算与更多运维约定见 `CLAUDE.md`

## 许可证

[MIT](LICENSE)

---

## 附录：从零搭建环境

实测环境：Windows 11 + WSL2（Ubuntu 22.04）+ Docker Desktop，RTX 4090 24GB。原生 Linux 亦可，跳过 WSL 部分。

### 1. NVIDIA 驱动

在 **Windows** 上安装最新 NVIDIA 驱动即可，WSL 内无需安装 CUDA。验证：

```bash
nvidia-smi    # 应列出显卡
```

### 2. WSL2

```powershell
# Windows PowerShell（管理员）
wsl --install -d Ubuntu-22.04
```

已是 WSL1 的用 `wsl --set-version Ubuntu-22.04 2` 转换。

### 3. Docker Desktop

1. 安装 Docker Desktop，保持默认 **WSL2 后端**
2. Settings → Resources → **WSL Integration** → 勾选你的发行版（如 Ubuntu-22.04）
3. 可选但推荐——把 Docker 数据盘迁出 C 盘（vLLM 镜像约 28GB）：Settings → Resources → Advanced → **Disk image location**，选其他盘目录

验证：

```bash
docker version
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 4. vLLM

无需手动安装——以官方镜像 `vllm/vllm-openai` 形式随 `docker compose up` 自动拉取。所有引擎参数在 `docker-compose.yml` 的 `command` 里调整。

### 5. Python

脚本和管理台需要 Python ≥3.10（Ubuntu 22.04 自带 `python3` 即可）。首次下载会自动安装 `modelscope` CLI，管理台首次启动会自建 venv 安装 FastAPI 等依赖。

### 磁盘位置约定

项目写入的所有内容（权重、缓存、日志）都在工作空间目录内；唯一的大额外占用是 Docker 镜像存储，在第 3.3 步处理。
