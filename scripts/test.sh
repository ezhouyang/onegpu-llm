#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

echo "== 1. 检查服务 =="
MODEL=$(curl -sf "${BASE_URL}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "served model: ${MODEL}"

echo "== 2. 对话测试 =="
curl -sf "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"用一句话介绍西安\"}],
    \"max_tokens\": 512
  }" | python3 -c "import sys,json; c=json.load(sys.stdin)['choices'][0]['message'].get('content') or ''; print(c.strip() or '(只有reasoning输出，属正常)')"

echo "== 3. 工具调用测试 =="
curl -sf "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"北京现在天气怎么样？请调用工具查询\"}],
    \"tools\": [{
      \"type\": \"function\",
      \"function\": {
        \"name\": \"get_weather\",
        \"description\": \"查询指定城市天气\",
        \"parameters\": {
          \"type\": \"object\",
          \"properties\": {\"city\": {\"type\": \"string\"}},
          \"required\": [\"city\"]
        }
      }
    }],
    \"tool_choice\": \"auto\",
    \"max_tokens\": 256
  }" | python3 -c "
import sys, json
msg = json.load(sys.stdin)['choices'][0]['message']
tc = msg.get('tool_calls')
if tc:
    print('tool_call:', tc[0]['function']['name'], tc[0]['function']['arguments'])
else:
    print('FAIL: 模型未返回 tool_calls'); sys.exit(1)
"

echo "全部通过"
