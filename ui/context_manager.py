"""上下文压缩：三层结构（系统契约层 / 压缩摘要层 / 工作记忆层）。

- 系统契约层（datetime、记忆、技能、搜索上下文）由 server 在每轮注入，不参与压缩
- 历史对话超预算时，最旧部分交由模型摘要，摘要以 chat_id 持久化，滚动累积
- 工作记忆层：最近 KEEP_RECENT 条消息原文
"""

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHATS_DIR = ROOT / "chats"
VLLM_PORT = 8000

KEEP_RECENT = 6
BUDGET_RATIO = 0.65
SUMMARY_MAX_CHARS = 800


def estimate_tokens(messages: list) -> int:
    """粗估：中英文混合约 2 字符/token。"""
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            total += sum(len(p.get("text", "")) for p in c if isinstance(p, dict))
    return total // 2


def _summary_path(chat_id: str) -> Path:
    return CHATS_DIR / f"_summary_{chat_id}.json"


def load_summary(chat_id: str | None) -> str:
    if not chat_id:
        return ""
    path = _summary_path(chat_id)
    if path.is_file():
        try:
            return json.loads(path.read_text()).get("summary", "")
        except Exception:
            pass
    return ""


def save_summary(chat_id: str, summary: str):
    if not chat_id:
        return
    CHATS_DIR.mkdir(exist_ok=True)
    _summary_path(chat_id).write_text(
        json.dumps({"summary": summary}, ensure_ascii=False)
    )


def _vllm_summarize(model: str, prompt: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "你是对话压缩器。输出简洁的客观摘要，不要寒暄。"},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 600,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{VLLM_PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return (data["choices"][0]["message"].get("content") or "").strip()


def compress_if_needed(chat_id: str | None, model: str, messages: list,
                       max_model_len: int) -> tuple[list, dict | None]:
    """超预算则压缩最旧历史。返回 (新消息列表, 压缩事件 or None)。"""
    budget = int(max_model_len * BUDGET_RATIO)
    if estimate_tokens(messages) <= budget:
        return messages, None
    if len(messages) <= KEEP_RECENT + 2:
        return messages, None

    old, recent = messages[:-KEEP_RECENT], messages[-KEEP_RECENT:]
    old_lines = []
    for m in old:
        role = m.get("role", "?")
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
        old_lines.append(f"{role}: {(c or '')[:500]}")
    prev_summary = load_summary(chat_id)

    prompt = ""
    if prev_summary:
        prompt += f"【已有摘要】\n{prev_summary}\n\n"
    prompt += (
        "【待压缩的历史对话】\n" + "\n".join(old_lines)
        + f"\n\n请把以上内容合并压缩成不超过{SUMMARY_MAX_CHARS}字的摘要，"
        "保留：用户原始目标与需求、关键事实与决定、未完成的待办。丢弃寒暄与过程性内容。"
    )
    try:
        summary = _vllm_summarize(model, prompt)
    except Exception:
        return messages, None
    save_summary(chat_id, summary)

    compressed = [{
        "role": "system",
        "content": f"（以下是此前 {len(old)} 条对话的压缩摘要，据此理解上下文）\n{summary}",
    }]
    return compressed + recent, {"compressed": len(old), "kept": len(recent)}
