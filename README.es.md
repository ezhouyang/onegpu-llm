<div align="center">
  <img src="ui/static/logo.svg" width="96" alt="onegpu-llm logo">
  <h1>onegpu-llm</h1>
  <p>Ejecuta LLMs de código abierto en una sola GPU de consumo — con una consola web para gestionarlo todo.</p>
  <p>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="Licencia: MIT"></a>
    <img src="https://img.shields.io/badge/engine-vLLM-green.svg" alt="Motor: vLLM">
    <img src="https://img.shields.io/badge/gpu-24GB%20VRAM-orange.svg" alt="GPU: 24GB VRAM">
  </p>
  <p><a href="README.md">English</a> · <a href="README.zh-CN.md">中文</a> · <b>Español</b></p>
</div>

---

## ¿Qué es esto?

Un espacio de trabajo listo para usar que despliega LLMs de código abierto (Qwen, Gemma, …) en una **sola GPU de 24GB** (probado en RTX 4090, WSL2 + Docker Desktop). La inferencia corre con **vLLM** exponiendo una API compatible con OpenAI, y una **consola web** ligera gestiona la descarga de modelos, el arranque/parada, los logs y la monitorización de la GPU.

Diseñado para cadenas de herramientas de IA personales: conecta el endpoint local a [opencode](https://opencode.ai), Cline, Continue o cualquier cliente compatible con OpenAI.

## Características

- **Consola web** (`http://localhost:9000`): descarga de modelos con progreso en vivo, arranque/parada con un clic, logs del contenedor, estadísticas de GPU en tiempo real (VRAM / utilización / temperatura / consumo), prueba de humo integrada (chat + llamadas a herramientas)
- **Panel de chat**: conversación multiturno con selector de modelo (usa el modelo en ejecución por defecto), renderizado Markdown/código, adjuntos de imagen y archivos de texto (compatible con modelos multimodales), **búsqueda web opcional** (DuckDuckGo gratuito, sin API key — los resultados se inyectan como contexto para reducir alucinaciones, con citas de fuentes)
- **Un modelo a la vez** por diseño: cambiar de modelo es un clic; la consola detiene el anterior automáticamente (24GB de VRAM no alcanzan para dos)
- **Registro de modelos cuantizados**: modelos AWQ ajustados a 24GB, con presupuesto de `max-model-len` por modelo
- **Llamadas a herramientas listas**: parser Hermes + auto tool choice, funciona con herramientas de coding agénticas
- **Reinicios rápidos**: la caché de compilación de vLLM se persiste en el workspace; los reinicios en caliente tardan ~70s
- **Todo permanece en el workspace**: pesos, cachés y logs — nada se escribe en `$HOME` ni en otras unidades

## Inicio rápido

Requisitos: GPU NVIDIA (≥24GB recomendado), Docker con soporte GPU (Docker Desktop + WSL2 funciona), Python 3.10+. La guía detallada de entorno está en el apéndice del [README en inglés](README.md#appendix-environment-setup-from-scratch).

```bash
git clone https://github.com/ezhouyang/onegpu-llm.git
cd onegpu-llm

./scripts/download.sh qwen3-14b          # descarga los pesos vía ModelScope
docker compose --profile qwen3-14b up -d # arranca el servicio de inferencia
./scripts/test.sh                        # prueba de humo: chat + tool calling
./scripts/ui.sh                          # consola web http://localhost:9000
```

La API compatible con OpenAI escucha en `http://localhost:8000/v1`.

## Registro de modelos

| Perfil | Modelo | Uso | VRAM (pesos) | max-model-len |
|---|---|---|---|---|
| `qwen3-14b` | Qwen/Qwen3-14B-AWQ | uso diario, chat/RAG en chino | ~9GB | 32768 |
| `qwen3-32b` | Qwen/Qwen3-32B-AWQ | máxima calidad en una GPU (lento, contexto justo) | ~19GB | 8192 |
| `qwen3-30b-a3b` | Qwen/Qwen3-30B-A3B-Instruct-2507-AWQ | MoE, rápido y casi calidad 32B | ~17GB | 16384 |
| `qwen25-coder-32b` | Qwen/Qwen2.5-Coder-32B-Instruct-AWQ | backend para agentes de código | ~19GB | 8192 |
| `gemma3-4b` | LLM-Research/gemma-3-4b-it | multimodal (entrada de imagen), ligero | ~8GB | 16384 |

Añadir un modelo = una línea en `scripts/download.sh` + un bloque de servicio en `docker-compose.yml`. Gemma 3 27B (requiere cuantización comunitaria) y otros modelos están planificados.

## ¿Por qué vLLM en lugar de Ollama?

| | onegpu-llm (vLLM) | Ollama |
|---|---|---|
| Rendimiento | PagedAttention + batching continuo, muy superior con concurrencia | peticiones en serie, batching limitado |
| Tool calling | de primera clase (`--tool-call-parser`), estable para agentes | varía según modelo/plantilla |
| Cuantización | AWQ/GPTQ/FP8 — mejor calidad que GGUF a igual VRAM | solo GGUF |
| Control fino | todos los parámetros del motor vía compose | pocas opciones |
| Facilidad | requiere Docker + algo de configuración (este repo lo resuelve) | instalación de una línea, ideal para pruebas rápidas |

Para un backend de agente de código de un solo usuario en una GPU, vLLM ofrece mejor calidad por GB de VRAM (AWQ) y tool calling más fiable; Ollama es más simple para experimentos.

## Integración con opencode

En `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "onegpu": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:8000/v1" },
      "models": {
        "qwen25-coder-32b": { "name": "Qwen2.5 Coder 32B (local)" },
        "qwen3-14b": { "name": "Qwen3 14B (local)" }
      }
    }
  },
  "model": "onegpu/qwen25-coder-32b"
}
```

Notas:

- El nombre del modelo servido lo define `--served-model-name` en `docker-compose.yml`; mantenlo sincronizado con las claves de `models`
- Arranca primero el perfil correspondiente; para programar se recomienda `qwen25-coder-32b`
- El mismo endpoint sirve para Cline / Continue / LibreChat: URL base OpenAI-compatible `http://localhost:8000/v1`, cualquier cadena como API key

## Estructura del proyecto

```
onegpu-llm/
├── docker-compose.yml     # servicios de modelos, un perfil por modelo
├── models/                # pesos (ignorado por git)
├── scripts/               # download.sh / test.sh / ui.sh
├── ui/                    # consola web (FastAPI + página única, sin build)
├── .cache/                # cachés modelscope/pip/vLLM (ignorado por git)
└── logs/                  # logs del servidor y descargas (ignorado por git)
```

## Notas

- Usuarios de WSL2: el model runner V2 de vLLM ≥0.25 requiere UVA, que WSL2 no soporta; este repo ya establece `VLLM_USE_V2_MODEL_RUNNER=0`
- Modo de razonamiento de Qwen3: añade `/no_think` al prompt si quieres respuestas directas
- Consulta `CLAUDE.md` para las convenciones operativas completas (presupuestos de VRAM, cómo añadir modelos)

## Licencia

[MIT](LICENSE)
