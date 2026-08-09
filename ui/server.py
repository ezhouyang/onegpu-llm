import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = ROOT / "docker-compose.yml"
MODELS_DIR = ROOT / "models"
VLLM_PORT = 8000

app = FastAPI(title="llm-deploy manager")

download_tasks: dict[str, dict] = {}


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
        rel = model_path.group(1).removeprefix("/models/") if model_path else ""
        local_path = MODELS_DIR / rel
        downloaded = local_path.is_dir() and any(local_path.rglob("*.safetensors"))
        models[key] = {
            "key": key,
            "container": svc.get("container_name", f"vllm-{key}"),
            "model_id": rel,
            "served_name": served.group(1) if served else key,
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


@app.post("/api/chat")
def chat(payload: dict):
    messages = payload.get("messages", [])
    if not messages:
        raise HTTPException(400, "messages required")
    health = vllm_health()
    if not health["ready"]:
        raise HTTPException(503, "没有正在运行的模型，请先启动")
    body = json.dumps({
        "model": health["served"][0],
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
    }


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
