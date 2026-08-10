import json
import re
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent_runtime import run_agent
from context_manager import compress_if_needed, load_summary
from mcp_manager import manager as mcp_manager
import memory_store
import skill_loader

ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = ROOT / "docker-compose.yml"
MODELS_DIR = ROOT / "models"
CHATS_DIR = ROOT / "chats"
VLLM_PORT = 8000

app = FastAPI(title="llm-deploy manager")

download_tasks: dict[str, dict] = {}


@app.on_event("startup")
def startup():
    if not (ROOT / "mcp.json").is_file() and (ROOT / "mcp.example.json").is_file():
        import shutil
        shutil.copy(ROOT / "mcp.example.json", ROOT / "mcp.json")
    try:
        mcp_manager.start_all()
    except Exception as e:
        print(f"[mcp] startup error: {e}")


@app.on_event("shutdown")
def shutdown():
    mcp_manager.shutdown()


def run(cmd: list[str], timeout: int = 30) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def load_models() -> dict[str, dict]:
    data = yaml.safe_load(COMPOSE_FILE.read_text())
    models = {}
    for svc_name, svc in data.get("services", {}).items():
        profiles = svc.get("profiles", [])
        key = profiles[0] if profiles else svc_name
        cmd = svc.get("command", "") or ""
        model_path = re.search(r"--model\s+(\S+)", cmd)
        served = re.search(r"--served-model-name\s+(\S+)", cmd)
        max_len = re.search(r"--max-model-len\s+(\d+)", cmd)
        util = re.search(r"--gpu-memory-utilization\s+([\d.]+)", cmd)
        rel = model_path.group(1).removeprefix("/models/") if model_path else ""
        local_path = MODELS_DIR / rel
        downloaded = local_path.is_dir() and any(local_path.rglob("*.safetensors"))
        models[key] = {
            "key": key,
            "container": svc.get("container_name", f"vllm-{key}"),
            "model_id": rel,
            "served_name": served.group(1) if served else key,
            "max_model_len": int(max_len.group(1)) if max_len else None,
            "gpu_memory_utilization": float(util.group(1)) if util else None,
            "tool_call": "--tool-call-parser" in cmd,
            "reasoning": "--reasoning-parser" in cmd,
            "multimodal": "--limit-mm-per-prompt" in cmd,
            "local_path": str(local_path),
            "downloaded": downloaded,
        }
    return models


def running_containers() -> list[str]:
    out = run(["docker", "ps", "--filter", "name=^vllm-", "--format", "{{.Names}}"])
    return [line for line in out.splitlines() if line]


def vllm_health() -> dict:
    try:
        with urllib.request.urlopen(f"http://localhost:{VLLM_PORT}/v1/models", timeout=2) as resp:
            data = json.loads(resp.read())
            return {"ready": True, "served": [m["id"] for m in data.get("data", [])]}
    except Exception:
        return {"ready": False, "served": []}


@app.get("/api/models")
def list_models():
    models = load_models()
    running = set(running_containers())
    health = vllm_health()
    for m in models.values():
        m["running"] = m["container"] in running
        m["api_ready"] = health["ready"] and m["served_name"] in health["served"]
        task = download_tasks.get(m["key"])
        m["downloading"] = bool(task and task["proc"].poll() is None)
    return {"models": list(models.values())}


@app.post("/api/models/{key}/start")
def start_model(key: str):
    models = load_models()
    if key not in models:
        raise HTTPException(404, f"unknown model: {key}")
    if not models[key]["downloaded"]:
        raise HTTPException(400, "model weights not downloaded, run scripts/download.sh first")
    for name in running_containers():
        if name != models[key]["container"]:
            run(["docker", "stop", name], timeout=180)
    try:
        out = run(["docker", "compose", "-f", str(COMPOSE_FILE), "--profile", key, "up", "-d"],
                  timeout=900)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return {"ok": True, "output": out}


@app.post("/api/models/{key}/stop")
def stop_model(key: str):
    models = load_models()
    if key not in models:
        raise HTTPException(404, f"unknown model: {key}")
    try:
        out = run(["docker", "compose", "-f", str(COMPOSE_FILE), "--profile", key, "down"],
                  timeout=180)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return {"ok": True, "output": out}


@app.get("/api/models/{key}/logs")
def model_logs(key: str, tail: int = 100):
    models = load_models()
    if key not in models:
        raise HTTPException(404, f"unknown model: {key}")
    container = models[key]["container"]
    result = subprocess.run(
        ["docker", "logs", "--tail", str(tail), container],
        capture_output=True, text=True, timeout=30,
    )
    text = (result.stdout + result.stderr).strip()
    if result.returncode != 0 and not text:
        text = f"(容器 {container} 不存在或未启动)"
    return {"logs": text}


@app.get("/api/models/{key}/info")
def model_info(key: str):
    models = load_models()
    if key not in models:
        raise HTTPException(404, f"unknown model: {key}")
    m = models[key]
    local = Path(m["local_path"])
    size_bytes = 0
    file_count = 0
    if local.is_dir():
        for f in local.rglob("*"):
            if f.is_file():
                size_bytes += f.stat().st_size
                file_count += 1
    running = set(running_containers())
    health = vllm_health()
    return {
        **m,
        "running": m["container"] in running,
        "api_ready": health["ready"] and m["served_name"] in health["served"],
        "disk_size_gb": round(size_bytes / 1024**3, 2),
        "file_count": file_count,
        "api_base": f"http://localhost:{VLLM_PORT}/v1" if health["ready"] else None,
    }


@app.post("/api/models")
def add_model(payload: dict):
    key = (payload.get("key") or "").strip()
    model_id = (payload.get("model_id") or "").strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", key):
        raise HTTPException(400, "key 只能包含小写字母、数字、连字符")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", model_id):
        raise HTTPException(400, "model_id 格式应为 org/name（ModelScope 模型 ID）")
    models = load_models()
    if key in models:
        raise HTTPException(400, f"profile {key} 已存在")
    if any(m["model_id"] == model_id for m in models.values()):
        raise HTTPException(400, f"模型 {model_id} 已存在")

    max_len = int(payload.get("max_model_len") or 8192)
    util = float(payload.get("gpu_memory_utilization") or 0.92)
    cmd_lines = [
        f"--model /models/{model_id}",
        f"--served-model-name {key}",
        f"--max-model-len {max_len}",
        f"--gpu-memory-utilization {util}",
    ]
    if payload.get("reasoning"):
        cmd_lines.append("--reasoning-parser deepseek_r1")
    if payload.get("tool_call"):
        cmd_lines.append("--tool-call-parser hermes\n      --enable-auto-tool-choice")
    if payload.get("multimodal"):
        cmd_lines.append("--limit-mm-per-prompt image=1")

    service_block = (
        f"\n  {key}:\n"
        f"    <<: *vllm-base\n"
        f"    container_name: vllm-{key}\n"
        f"    profiles: [\"{key}\"]\n"
        f"    command: >\n"
        + "".join(f"      {line}\n" for line in cmd_lines)
    )
    with open(COMPOSE_FILE, "a") as f:
        f.write(service_block)

    download_sh = ROOT / "scripts" / "download.sh"
    lines = download_sh.read_text().splitlines(keepends=True)
    entry = f'  [{key}]="{model_id}"\n'
    for i, line in enumerate(lines):
        if line.rstrip() == ")" and any("declare -A MODELS" in l for l in lines[:i]):
            lines.insert(i, entry)
            break
    else:
        raise HTTPException(500, "无法在 download.sh 中定位模型表")
    download_sh.write_text("".join(lines))

    return {"ok": True, "key": key, "model_id": model_id}


@app.post("/api/models/{key}/download")
def download_model(key: str):
    models = load_models()
    if key not in models:
        raise HTTPException(404, f"unknown model: {key}")
    if models[key]["downloaded"]:
        return {"ok": True, "status": "already_downloaded"}
    task = download_tasks.get(key)
    if task and task["proc"].poll() is None:
        return {"ok": True, "status": "downloading"}
    log_path = ROOT / "logs" / f"download-{key}.log"
    log_path.parent.mkdir(exist_ok=True)
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        ["bash", str(ROOT / "scripts" / "download.sh"), key],
        stdout=log_file, stderr=subprocess.STDOUT,
    )
    download_tasks[key] = {"proc": proc, "log": str(log_path), "started": time.time()}
    return {"ok": True, "status": "started"}


@app.get("/api/models/{key}/download-status")
def download_status(key: str):
    task = download_tasks.get(key)
    if not task:
        return {"status": "idle", "log": ""}
    raw = Path(task["log"]).read_text(errors="replace")
    lines = [ln for ln in raw.replace("\r", "\n").splitlines() if ln.strip()]
    log_text = "\n".join(lines[-30:])
    code = task["proc"].poll()
    if code is None:
        status = "downloading"
    elif code == 0:
        status = "done"
    else:
        status = "failed"
    return {"status": status, "log": log_text}


def web_search(query: str, max_results: int = 5) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        raise HTTPException(503, "搜索依赖未安装，请运行: ui/.venv/bin/pip install ddgs")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        raise HTTPException(502, f"搜索失败: {e}")
    return results


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def build_chat_request(payload: dict) -> tuple[str, list, list, dict | None]:
    messages = payload.get("messages", [])
    if not messages:
        raise HTTPException(400, "messages required")
    health = vllm_health()
    if not health["ready"]:
        raise HTTPException(503, "没有正在运行的模型，请先启动")
    model = payload.get("model") or health["served"][0]
    if model not in health["served"]:
        raise HTTPException(503, f"模型 {model} 未在运行，请先在控制台启动")

    is_agent = bool(payload.get("_agent"))
    chat_id = payload.get("chat_id")
    models_cfg = load_models()
    running_cfg = next((m for m in models_cfg.values() if m["served_name"] == model), {})
    max_len = running_cfg.get("max_model_len") or 8192

    compress_event = None
    messages, compress_event = compress_if_needed(chat_id, model, messages, max_len)

    system_parts = [
        f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')}。"
        "涉及「今天」「现在」「最近」等时间相关问题时，以此时间为准。"
    ]
    mem = memory_store.digest(extract_text(messages[-1].get("content", "")))
    if mem:
        system_parts.append("以下是你记住的关于用户的信息：\n" + mem)
    if is_agent:
        skills = skill_loader.digest()
        if skills:
            system_parts.append(skills)
        system_parts.append(
            "你可以使用 memory__save 工具把值得长期记住的信息（用户偏好、重要事实、任务结论）保存下来。"
        )
    prev_summary = load_summary(chat_id)
    if prev_summary and not compress_event:
        # 摘要已在压缩时注入；未触发新压缩但存在历史摘要时补入
        if not any(prev_summary[:50] in str(m.get("content", "")) for m in messages[:1]):
            system_parts.append(f"（此前对话的压缩摘要）\n{prev_summary}")

    search_results = []
    if payload.get("web_search"):
        query = extract_text(messages[-1].get("content", ""))
        search_results = web_search(query)
        context = "\n".join(
            f"[{i+1}] {r.get('title','')}\n{r.get('body','')}\n来源: {r.get('href','')}"
            for i, r in enumerate(search_results)
        )
        system_parts.append(
            "以下是与用户问题相关的最新网络搜索结果。请优先结合搜索结果回答，"
            "回答时在引用处标注来源编号（如 [1]），并在末尾列出「参考来源」清单。"
            "如果搜索结果与问题无关，直接忽略它们，也不要列出参考来源。\n\n" + context
        )
    messages = [{"role": "system", "content": "\n\n".join(system_parts)}] + messages
    return model, messages, search_results, compress_event


def slim_results(results: list[dict]) -> list[dict]:
    return [
        {"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")}
        for r in results
    ]


@app.get("/api/memory")
def memory_list():
    return {"entries": memory_store.load()}


@app.post("/api/memory")
def memory_add(payload: dict):
    try:
        entry = memory_store.add(payload.get("content", ""), payload.get("type", "fact"), "manual")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "entry": entry}


@app.post("/api/memory/{entry_id}/delete")
def memory_delete(entry_id: str):
    if not memory_store.remove(entry_id):
        raise HTTPException(404, "entry not found")
    return {"ok": True}


@app.get("/api/skills")
def skills_list():
    return {"skills": [{k: s[k] for k in ("name", "description")} for s in skill_loader.scan()]}


@app.get("/api/skills/{name}")
def skill_read(name: str):
    try:
        content = skill_loader.read(name)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {"name": name, "content": content}


@app.post("/api/skills")
def skill_write(payload: dict):
    name = (payload.get("name") or "").strip()
    description = (payload.get("description") or "").strip()
    content = payload.get("content") or ""
    if not name or not description:
        raise HTTPException(400, "name 和 description 必填")
    try:
        skill_loader.write(name, description, content)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "name": name}


@app.get("/api/mcp")
def mcp_status():
    servers = []
    for name, cfg in mcp_manager.configs.items():
        servers.append({
            "name": name,
            "type": cfg.get("type", "local"),
            "status": mcp_manager.statuses.get(name, "unknown"),
            "error": mcp_manager.errors.get(name),
            "enabled": cfg.get("enabled", True),
            "command_or_url": " ".join(cfg.get("command", [])) or cfg.get("url", ""),
            "tools": mcp_manager.tools.get(name, []),
        })
    return {"servers": servers}


@app.post("/api/mcp/reload")
def mcp_reload():
    try:
        mcp_manager.start_all()
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"ok": True}


def _load_mcp_config() -> dict:
    path = ROOT / "mcp.json"
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"servers": {}}


def _save_mcp_config(data: dict):
    (ROOT / "mcp.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))


@app.post("/api/mcp/servers")
def mcp_add_server(payload: dict):
    name = (payload.get("name") or "").strip()
    stype = payload.get("type", "local")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise HTTPException(400, "name 只能包含小写字母、数字、连字符、下划线")
    data = _load_mcp_config()
    if name in data["servers"]:
        raise HTTPException(400, f"server {name} 已存在")
    if stype == "local":
        command = payload.get("command", [])
        if isinstance(command, str):
            import shlex
            command = shlex.split(command)
        if not command:
            raise HTTPException(400, "local 类型需要 command")
        entry = {"type": "local", "command": command,
                 "enabled": payload.get("enabled", True), "env": payload.get("env") or {}}
    else:
        url = (payload.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            raise HTTPException(400, "remote 类型需要 http(s) url")
        entry = {"type": "remote", "url": url, "enabled": payload.get("enabled", True)}
    data["servers"][name] = entry
    _save_mcp_config(data)
    mcp_reload()
    return {"ok": True, "name": name}


@app.post("/api/mcp/servers/{name}/toggle")
def mcp_toggle_server(name: str):
    data = _load_mcp_config()
    if name not in data["servers"]:
        raise HTTPException(404, f"unknown server: {name}")
    data["servers"][name]["enabled"] = not data["servers"][name].get("enabled", True)
    _save_mcp_config(data)
    mcp_reload()
    return {"ok": True, "enabled": data["servers"][name]["enabled"]}


@app.post("/api/chat/agent")
def chat_agent(payload: dict):
    payload["_agent"] = True

    def gen():
        try:
            model, messages, _, compress_event = build_chat_request(payload)
        except HTTPException as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e.detail)}, ensure_ascii=False)}\n\n"
            return
        if compress_event:
            text = f"已压缩 {compress_event['compressed']} 条历史对话（保留最近 {compress_event['kept']} 条原文）"
            yield f"data: {json.dumps({'type': 'notice', 'text': text}, ensure_ascii=False)}\n\n"

        import queue
        import threading

        q: queue.Queue = queue.Queue()

        def work():
            result = run_agent(model, messages, mcp_manager, q.put)
            q.put({"type": "final", "message": result})

        threading.Thread(target=work, daemon=True).start()
        while True:
            ev = q.get()
            yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
            if ev.get("type") in ("final", "error"):
                break

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/chat/stream")
def chat_stream(payload: dict):
    def gen():
        try:
            model, messages, search_results, compress_event = build_chat_request(payload)
        except HTTPException as e:
            yield f"data: {json.dumps({'error': str(e.detail)})}\n\n"
            return
        if compress_event:
            ctext = f"已压缩 {compress_event['compressed']} 条历史对话（保留最近 {compress_event['kept']} 条原文）"
            yield f"data: {json.dumps({'notice': ctext}, ensure_ascii=False)}\n\n"
        if search_results:
            yield f"data: {json.dumps({'search_results': slim_results(search_results)}, ensure_ascii=False)}\n\n"
        body = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": 2048,
            "stream": True,
            "stream_options": {"include_usage": True},
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line.startswith("data:"):
                        yield line + "\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': f'vLLM 请求失败: {e}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/chat")
def chat(payload: dict):
    model, messages, search_results, _ = build_chat_request(payload)
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": 2048,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{VLLM_PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        raise HTTPException(502, f"vLLM 请求失败: {e}")
    msg = data["choices"][0]["message"]
    return {
        "content": (msg.get("content") or "").strip(),
        "reasoning": (msg.get("reasoning") or "").strip(),
        "search_results": slim_results(search_results),
    }


CHATS_DIR.mkdir(exist_ok=True)


@app.get("/api/chats")
def list_chats():
    chats = []
    for f in sorted(CHATS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            d = json.loads(f.read_text(errors="replace"))
            chats.append({
                "id": d.get("id", f.stem),
                "title": d.get("title", f.stem),
                "updated": d.get("updated", 0),
                "model": d.get("model", ""),
            })
        except Exception:
            continue
    return {"chats": chats}


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str):
    path = CHATS_DIR / f"{chat_id}.json"
    if not path.is_file():
        raise HTTPException(404, "chat not found")
    return json.loads(path.read_text(errors="replace"))


@app.post("/api/chats")
def save_chat(payload: dict):
    chat_id = payload.get("id") or datetime.now().strftime("%Y%m%d-%H%M%S")
    messages = payload.get("messages", [])
    title = payload.get("title") or next(
        (extract_text(m.get("content", ""))[:30] for m in messages if m.get("role") == "user"),
        "新对话",
    )
    model = next(
        (m.get("metrics", {}).get("model", "") for m in reversed(messages)
         if m.get("role") == "assistant" and m.get("metrics")),
        "",
    )
    record = {
        "id": chat_id,
        "title": title,
        "model": model,
        "updated": time.time(),
        "messages": messages,
    }
    (CHATS_DIR / f"{chat_id}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return {"id": chat_id, "title": title}


@app.post("/api/test")
def smoke_test():
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "test.sh")],
        capture_output=True, text=True, timeout=120,
    )
    output = (result.stdout + result.stderr).strip()
    return {"ok": result.returncode == 0, "output": output}


@app.get("/api/gpu")
def gpu_status():
    out = run([
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ])
    name, util, mem_used, mem_total, temp, power = [p.strip() for p in out.splitlines()[0].split(",")]
    return {
        "name": name,
        "utilization": int(util),
        "memory_used": int(mem_used),
        "memory_total": int(mem_total),
        "temperature": int(temp),
        "power": float(power),
    }


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
