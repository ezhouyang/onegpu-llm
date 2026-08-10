"""Agent 运行循环：vLLM 推理 + MCP 工具调用迭代。"""

import json
import time
import urllib.request

VLLM_PORT = 8000
MAX_ITERATIONS = 5
TOOL_RESULT_LIMIT = 4000


def vllm_chat(model: str, messages: list, tools: list | None) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 2048,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    req = urllib.request.Request(
        f"http://localhost:{VLLM_PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def run_agent(model: str, messages: list, mcp_manager, emit, max_iterations: int = MAX_ITERATIONS):
    """执行 agent 循环。emit(event_dict) 推送 SSE 事件。返回最终 assistant 消息。"""
    tools, mapping = mcp_manager.all_tools_openai()
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    iterations = 0
    t0 = time.time()

    working = list(messages)
    while True:
        iterations += 1
        if iterations > max_iterations:
            emit({"type": "notice", "text": f"达到最大迭代数 {max_iterations}，停止工具调用"})
            tools = None

        try:
            data = vllm_chat(model, working, tools or None)
        except Exception as e:
            emit({"type": "error", "text": f"vLLM 请求失败: {e}"})
            return None

        usage = data.get("usage") or {}
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)

        choice = data["choices"][0]
        msg = choice["message"]
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            metrics = {
                "iterations": iterations,
                "total": f"{time.time() - t0:.1f}",
                "tokens": total_usage["completion_tokens"],
                "model": model,
            }
            return {
                "role": "assistant",
                "content": (msg.get("content") or "").strip(),
                "reasoning": (msg.get("reasoning") or "").strip(),
                "metrics": metrics,
            }

        working.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            fn = tc.get("function", {})
            full_name = fn.get("name", "")
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}
                raw_args = "{}"
            emit({"type": "tool_call", "name": full_name, "arguments": raw_args})

            server, tool = mapping.get(full_name, (None, None))
            if server is None:
                result_text = f"错误：未知工具 {full_name}"
            else:
                try:
                    result_text = mcp_manager.call_tool(server, tool, args)
                except Exception as e:
                    result_text = f"工具执行失败: {e}"
            if len(result_text) > TOOL_RESULT_LIMIT:
                result_text = result_text[:TOOL_RESULT_LIMIT] + "\n...(结果已截断)"
            emit({"type": "tool_result", "name": full_name, "preview": result_text[:500]})

            working.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result_text,
            })
