# Changelog

本项目变更记录。格式：[版本] - 日期，分类：Added / Changed / Fixed。

## [Unreleased]

### Added
- 管理台「+ 添加模型」表单：填写 profile / ModelScope ID / 显存与上下文参数 / 能力开关，自动写入 docker-compose.yml 和 download.sh
- 模型卡片新增「对话」（打开对话面板并选中该模型）和「信息」（本地路径、磁盘占用、参数、API 地址等）按钮
- 新增模型 qwen3-8b（轻量档）

### Fixed
- 对话历史下拉框刷新后不加载的真正修复（编辑落位错误，初始化改在 DOMContentLoaded）

## [0.2.0] - 2026-08-10

### Added
- 对话面板：多轮对话、模型选择器（默认当前运行模型）、Markdown/代码高亮渲染、图片与文本附件（多模态）
- 联网搜索：DuckDuckGo（ddgs，免费无 Key），搜索结果注入上下文并附引用；Kimi 式交互（生成中展开、完成后自动折叠、可手动展开）
- 系统时间注入 system prompt，解决「今天星期几」类问题
- SSE 流式输出（`/api/chat/stream`），每条回复记录 TTFT / tok/s / tokens / 总耗时，随消息保存供模型对比
- 对话持久化：`chats/<id>.json`（不进 git），下拉切换历史对话、新建对话
- 回复与代码块拷贝按钮
- 新增模型 gemma3-4b（多模态，LLM-Research/gemma-3-4b-it）
- vLLM 编译缓存挂载 `.cache/vllm`，热重启约 70s
- README 三语文档（EN/中文/ES 互链）、环境搭建附录、vs Ollama 对比、opencode 接入案例
- 存储位置约定：模型/缓存/日志/对话全部在工作空间内，避开 C 盘

### Fixed
- WSL2 下 vLLM V2 runner UVA 报错（`VLLM_USE_V2_MODEL_RUNNER=0`）
- modelscope 下载进度 `\r` 导致日志显示混乱
- 对话历史下拉框刷新后不加载（脚本执行顺序，改 DOMContentLoaded 触发）
- Markdown 渲染双重空行（气泡 pre-wrap 污染）、marked 开启 breaks/gfm、高亮改为完成后一次性执行

## [0.1.0] - 2026-08-09

### Added
- 初始版本：docker-compose 多 profile 模型服务（qwen3-14b / qwen3-32b / qwen3-30b-a3b / qwen25-coder-32b，AWQ 量化，统一 8000 端口）
- scripts：download.sh（ModelScope）、test.sh（冒烟测试：对话 + 工具调用）、ui.sh
- Web 管理台 v1：模型卡片、一键启停（自动单模型切换）、容器日志、GPU 实时监控、冒烟测试
- 管理台内模型下载（后台任务 + 实时进度日志）
- 项目开源：MIT LICENSE、README、logo，发布至 GitHub
