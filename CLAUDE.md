# onegpu-llm

本地 GPU（RTX 4090 24GB, WSL2）上的开源大模型部署工作空间。推理引擎统一用 **vLLM**（Docker 镜像 `vllm/vllm-openai`），对外提供 OpenAI 兼容 API，供 opencode 等工具接入。

## 变更与进度管理

- `CHANGELOG.md`：所有功能变更、修复必须登记（分类 Added/Changed/Fixed），提交前更新
- `ROADMAP.md`：待办按 P0/P1/P2 分级，完成一项移入 CHANGELOG；新想法先记这里再排期
- 版本号约定：功能批次升 minor（0.x.0），修复升 patch

## 目录约定

```
llm-deploy/
├── CLAUDE.md              # 本文件，先改这里再改实践
├── docker-compose.yml     # 所有模型服务定义，按 profile 隔离
├── .gitignore
├── models/                # 模型权重，不进 git
│   └── <org>/<name>/      # 目录结构镜像 ModelScope/HF 的模型 ID，如 Qwen/Qwen3-14B-AWQ
├── scripts/
│   ├── download.sh        # 模型下载（ModelScope，国内速度快）
│   ├── test.sh            # 冒烟测试（对话 + 工具调用）
│   └── ui.sh              # 启动管理界面
├── ui/                    # 可视化管理界面（FastAPI + 单页面，无构建步骤）
│   ├── server.py          # 后端：读 compose 配置、控制 docker、采集 GPU 状态、管理下载任务
│   ├── static/index.html  # 前端：模型卡片 + 下载/启停 + 日志 + GPU 监控
│   └── requirements.txt
├── .cache/                # modelscope/pip/tmp 缓存，不进 git
├── chats/                 # 对话记录（每对话一个 JSON），不进 git
└── logs/                  # vLLM 及下载日志，不进 git
```

## 管理界面

- 启动：`./scripts/ui.sh`，浏览器访问 `http://localhost:9000`
- 依赖装在 `ui/.venv`，不污染全局 Python 环境
- 后端只做薄封装：模型信息以 `docker-compose.yml` 为唯一数据源（解析 profiles 和 command），不在 ui 里维护重复配置
- 对话面板：模型选择默认当前运行模型；联网搜索用 DuckDuckGo（`ddgs` 包，免费无 Key），搜索结果注入 system prompt 让模型引用；图片附件以 base64 走 OpenAI 多模态格式，仅多模态模型可用
- 对话走 SSE 流式（`/api/chat/stream`），记录 TTFT/tok/s 等指标随消息保存，供模型间比较
- 对话记录持久化在 `chats/`（每对话一个 JSON），不进 git
- vLLM 编译缓存挂载在 `./.cache/vllm`，不要删（删了重启会重新编译，变慢）

## 存储位置约定（全部避开 C 盘）

- 模型权重：`models/<org>/<name>/`（D 盘工作空间内）
- 各类缓存：`.cache/modelscope`、`.cache/pip`、`.cache/tmp`（工作空间内，由脚本设置 `MODELSCOPE_CACHE` / `PIP_CACHE_DIR` / `TMPDIR`）
- 下载/运行日志：`logs/`（工作空间内）
- Docker 镜像与容器层：建议将 Docker Desktop 数据盘迁出 C 盘（settings-store.json 的 `CustomWslDistroDir` 指向其他盘），vLLM 镜像约 28GB
- 新增任何会写盘的工具，先把缓存/数据目录指到工作空间，再使用

## 硬性约束（由 24GB 显存决定）

1. **同一时间只跑一个模型**。切换模型 = 停掉当前 profile，启动另一个。
2. **14B 以上模型必须用量化版**（优先官方 AWQ，其次社区 AWQ/GPTQ/FP8）。FP16 的 14B 就要 ~28GB，放不下。
3. 所有服务统一监听 `8000` 端口，由「单模型运行」约束保证不冲突。
4. 新增模型：优先用管理台「+ 添加模型」表单（自动写入 compose 和 download.sh）；手动方式 = download.sh 模型表加一行 + compose 复制一个 service
5. 模型 ID 变动频繁（尤其社区量化版），下载失败时先去 modelscope.cn 搜同名模型确认 ID。

## 显存预算参考（4090 24GB，`--gpu-memory-utilization 0.92` ≈ 22.5GB 可用）

| 权重占用 | KV cache 余量 | max-model-len 建议 |
|---|---|---|
| ~9GB（14B AWQ） | ~13GB | 32768 |
| ~17GB（30B MoE AWQ） | ~5GB | 16384 |
| ~19GB（32B AWQ） | ~3GB | 8192 |

新模型按「权重每 +2GB，max-model-len 减半」粗估，启动后看 vLLM 日志里 `Maximum concurrency` 一行再微调。

## 模型清单

| profile | 模型 | 用途 | 状态 |
|---|---|---|---|
| qwen3-14b | Qwen/Qwen3-14B-AWQ | 主力，中文对话/RAG | 待部署 |
| qwen3-32b | Qwen/Qwen3-32B-AWQ | 单机质量上限（慢，上下文紧） | 待部署 |
| qwen3-30b-a3b | Qwen/Qwen3-30B-A3B-Instruct-2507-AWQ | MoE，速度快质量接近32B | 待部署 |
| qwen25-coder-32b | Qwen/Qwen2.5-Coder-32B-Instruct-AWQ | coding 专用，接 opencode | 待部署 |
| gemma3-4b | LLM-Research/gemma-3-4b-it | 多模态（支持图片输入），轻量 | 待部署 |
| gemma3-27b | （待定，需社区量化版） | 视觉/英文 | 规划中 |

## vLLM 参数约定

- 支持思考模式的 Qwen3：加 `--reasoning-parser deepseek_r1`（关闭思考可用 `--enable-reasoning` 与否按版本而定，对话时也可用 `/no_think`）
- Qwen2.5 系列无思考模式，不加 reasoning 参数
- 需要被 opencode 等 agent 调用的模型：必须加 `--tool-call-parser hermes --enable-auto-tool-choice`
- Gemma 3 多模态：需加 `--limit-mm-per-prompt image=1` 控制显存

## 常用命令

```bash
./scripts/download.sh qwen3-14b                  # 下载模型
docker compose --profile qwen3-14b up -d         # 启动
docker compose --profile qwen3-14b down          # 停止
./scripts/test.sh                                # 冒烟测试
docker compose --profile qwen3-14b logs -f       # 看日志
```

## 验证要求

改完 compose 或新增模型后，必须跑 `./scripts/test.sh`（对话 + 工具调用都通过才算完成），不能只改不验。

## 红线

- 模型权重不进 git、不进 commit
- 不删 `models/` 下任何目录（删模型前先问）
- API Key 等凭据如需引入，走环境变量，不写进 compose 文件
