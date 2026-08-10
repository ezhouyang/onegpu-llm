"""长期记忆存储：memory/memory.json，三类记忆，容量有界。

写入方式：agent 模式下模型通过内置工具 memory__save 主动保存。
检索：最近 N 条 + 与当前问题关键词匹配的条目，注入 system prompt。
"""

import json
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = ROOT / "memory"
MEMORY_FILE = MEMORY_DIR / "memory.json"

MAX_ENTRIES = 200
DIGEST_MAX_CHARS = 1200
VALID_TYPES = {"preference", "fact", "conclusion"}


def load() -> list:
    if MEMORY_FILE.is_file():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except Exception:
            pass
    return []


def save(entries: list):
    MEMORY_DIR.mkdir(exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(entries[-MAX_ENTRIES:], ensure_ascii=False, indent=2))


def add(content: str, mtype: str = "fact", source: str = "") -> dict:
    content = content.strip()[:500]
    if not content:
        raise ValueError("empty content")
    if mtype not in VALID_TYPES:
        mtype = "fact"
    entries = load()
    entry = {
        "id": uuid.uuid4().hex[:8],
        "type": mtype,
        "content": content,
        "created": time.time(),
        "source": source,
    }
    entries.append(entry)
    save(entries)
    return entry


def remove(entry_id: str) -> bool:
    entries = load()
    remaining = [e for e in entries if e["id"] != entry_id]
    if len(remaining) == len(entries):
        return False
    save(remaining)
    return True


def digest(query: str = "") -> str:
    """最近 10 条 + 关键词命中，控制在 DIGEST_MAX_CHARS 内。"""
    entries = load()
    if not entries:
        return ""
    picked = entries[-10:]
    if query:
        keywords = {w for w in query if ord(w) > 127}  # 中文字符逐字匹配
        words = set(query.lower().split())
        for e in entries[:-10]:
            text = e["content"].lower()
            if (words and any(w in text for w in words if len(w) > 2)) or \
               (keywords and sum(1 for w in keywords if w in e["content"]) >= 3):
                if e not in picked:
                    picked.insert(0, e)
    type_label = {"preference": "偏好", "fact": "事实", "conclusion": "结论"}
    lines = [f"- [{type_label.get(e['type'], '事实')}] {e['content']}" for e in picked]
    out = []
    total = 0
    for line in lines:
        if total + len(line) > DIGEST_MAX_CHARS:
            break
        out.append(line)
        total += len(line)
    return "\n".join(out)
