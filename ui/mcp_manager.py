"""MCP server 连接管理：stdio/SSE 常驻连接、工具发现、工具调用。

在独立 asyncio 事件循环线程中运行，对外提供同步接口。
单个 server 故障只标记降级，不影响其他 server。
"""

import asyncio
import threading
from contextlib import AsyncExitStack
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "mcp.json"


class MCPManager:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()
        self.stack: AsyncExitStack | None = None
        self.sessions = {}   # name -> ClientSession
        self.statuses = {}   # name -> "connecting" | "ok" | "failed" | "disabled"
        self.errors = {}     # name -> error message
        self.tools = {}      # name -> [ {name, description, inputSchema} ]
        self.configs = {}    # name -> raw config

    def run(self, coro, timeout=60):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    # ---- 生命周期 ----

    def start_all(self, timeout=90):
        self.run(self._start_all(), timeout)

    async def _start_all(self):
        await self._shutdown_all()
        self.stack = AsyncExitStack()
        await self.stack.__aenter__()
        import json
        if CONFIG_PATH.is_file():
            try:
                data = json.loads(CONFIG_PATH.read_text())
                self.configs = data.get("servers", {})
            except Exception as e:
                self.configs = {}
                self.errors["(config)"] = str(e)
        for name, cfg in self.configs.items():
            if not cfg.get("enabled", True):
                self.statuses[name] = "disabled"
                continue
            self.statuses[name] = "connecting"
            try:
                await self._connect(name, cfg)
                self.statuses[name] = "ok"
                self.errors.pop(name, None)
            except Exception as e:
                self.statuses[name] = "failed"
                self.errors[name] = str(e)

    async def _connect(self, name, cfg):
        from mcp import ClientSession, StdioServerParameters
        stype = cfg.get("type", "local")
        if stype == "local":
            from mcp.client.stdio import stdio_client
            command = cfg["command"]
            env = None
            if cfg.get("env"):
                import os
                env = {**os.environ, **cfg["env"]}
            params = StdioServerParameters(command=command[0], args=command[1:], env=env)
            read, write = await self.stack.enter_async_context(stdio_client(params))
        else:
            from mcp.client.sse import sse_client
            read, write = await self.stack.enter_async_context(sse_client(cfg["url"]))
        session = await self.stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        result = await session.list_tools()
        self.sessions[name] = session
        self.tools[name] = [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema or {},
            }
            for t in result.tools
        ]

    async def _shutdown_all(self):
        if self.stack is not None:
            try:
                await self.stack.aclose()
            except Exception:
                pass
        self.stack = None
        self.sessions = {}
        self.statuses = {}
        self.tools = {}

    def shutdown(self):
        if not self.loop.is_closed():
            self.run(self._shutdown_all(), timeout=15)

    # ---- 工具 ----

    def all_tools_openai(self, max_tools: int = 10) -> tuple[list, dict]:
        """返回 (OpenAI tools 格式列表, 带命名空间的名称 -> (server, tool) 映射)"""
        tools = []
        mapping = {}
        for server, tool_list in self.tools.items():
            if self.statuses.get(server) != "ok":
                continue
            for t in tool_list:
                if len(tools) >= max_tools:
                    return tools, mapping
                full_name = f"{server}__{t['name']}"
                tools.append({
                    "type": "function",
                    "function": {
                        "name": full_name,
                        "description": t["description"] or t["name"],
                        "parameters": t["inputSchema"] or {"type": "object", "properties": {}},
                    },
                })
                mapping[full_name] = (server, t["name"])
        return tools, mapping

    def call_tool(self, server: str, tool: str, arguments: dict, timeout=30) -> str:
        return self.run(self._call_tool(server, tool, arguments), timeout)

    async def _call_tool(self, server: str, tool: str, arguments: dict) -> str:
        session = self.sessions.get(server)
        if session is None:
            raise RuntimeError(f"MCP server {server} 未连接")
        result = await session.call_tool(tool, arguments)
        parts = []
        for block in result.content:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(text)
            else:
                parts.append(str(block))
        return "\n".join(parts) or "(空结果)"


manager = MCPManager()
