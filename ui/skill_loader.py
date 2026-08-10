"""技能加载：skills/<name>/SKILL.md，渐进披露。

平时只注入 name + description；agent 模式下模型用内置工具 skills__read 读取全文。
frontmatter 用简单 --- 分隔的 key: value 格式（与 opencode/Claude SKILL.md 兼容）。
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT / "skills"


def scan() -> list:
    skills = []
    if not SKILLS_DIR.is_dir():
        return skills
    for path in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        name = path.parent.name
        description = ""
        m = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
        if m:
            fm = m.group(1)
            n = re.search(r"^name:\s*(.+)$", fm, re.MULTILINE)
            d = re.search(r"^description:\s*(.+)$", fm, re.MULTILINE)
            if n:
                name = n.group(1).strip()
            if d:
                description = d.group(1).strip()
        skills.append({"name": name, "description": description, "path": str(path)})
    return skills


def digest() -> str:
    skills = scan()
    if not skills:
        return ""
    lines = [f"- {s['name']}: {s['description']}" for s in skills if s["description"]]
    if not lines:
        return ""
    return "可用技能（需要时用 skills__read 工具读取全文）：\n" + "\n".join(lines)


def read(name: str) -> str:
    for s in scan():
        if s["name"] == name:
            text = Path(s["path"]).read_text(errors="replace")
            m = re.match(r"^---\s*\n.*?\n---\s*\n", text, re.DOTALL)
            body = text[m.end():] if m else text
            return body.strip()[:8000]
    raise KeyError(f"技能不存在: {name}")


def write(name: str, description: str, content: str):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise ValueError("技能名只能包含小写字母、数字、连字符")
    folder = SKILLS_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{content}\n"
    )
