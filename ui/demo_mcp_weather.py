"""示例 MCP server：天气查询（wttr.in，免费无需 API Key）。

stdio 传输，由 ui/mcp_manager.py 以子进程方式拉起。
"""

import json
import urllib.parse
import urllib.request

from fastmcp import FastMCP

mcp = FastMCP("weather")


@mcp.tool()
def get_weather(city: str) -> str:
    """查询指定城市的实时天气，返回温度、体感、湿度、风速、天气描述。

    Args:
        city: 城市名，中英文均可，如 "北京" / "Hangzhou"
    """
    url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1&lang=zh"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        d = json.loads(resp.read())
    cur = d["current_condition"][0]
    return json.dumps({
        "城市": city,
        "天气": cur["lang_zh"][0]["value"] if cur.get("lang_zh") else cur["weatherDesc"][0]["value"],
        "温度°C": cur["temp_C"],
        "体感°C": cur["FeelsLikeC"],
        "湿度%": cur["humidity"],
        "风速km/h": cur["windspeedKmph"],
    }, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
