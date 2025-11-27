# LLM Server  
### A Production-Style LLM API Gateway & Inference Runtime

This project is a self-hosted, production-inspired **LLM serving platform** built with FastAPI, Hugging Face Transformers, and PyTorch.

It is intentionally designed to mirror real-world architecture patterns used by companies deploying foundation models behind internal and external APIs.

Rather than being just a demo model runner, this repository focuses on:

- Infrastructure
- Security
- Quotas
- Observability
- Scalability patterns
- Clean separation of concerns
- Multi-model support

It is designed to be a **portfolio-grade system-level project** demonstrating my ability to design and build ML/AI infrastructure.

---

## Key Features

- FastAPI-based LLM gateway
- Multi-model routing (`MultiModelManager`)
- API key authentication
- Rate limiting & concurrency control
- Usage quotas (monthly limits)
- Prometheus metrics + Grafana-ready
- SQLAlchemy + Alembic for persistence
- Streaming & non-streaming generation
- Hugging Face Transformers backend
- Local + Remote model support
- Model discovery endpoint (`/v1/models`)
- Completion cache (deduplication)
- Evals framework (GSM8K, MMLU, MBPP, summarization, toxicity)
- Docker + `uv` support
- Test suite with Pytest
- Configurable for CPU / Apple Silicon (MPS) / CUDA
- Admin and Ops API
- Token counting

---

## Multi-Model Support

The system supports multiple models via a **MultiModelManager** and routing layer.

Models can be:

- **Local** (loaded via Hugging Face in-process)
- **Remote** (called via HTTP using `HttpLLMClient`)

Routing is controlled by a `models.yaml` file or environment variables.

Example `models.yaml`:

~~~yaml
default_model: mistralai/Mistral-7B-v0.1

models:
  - id: mistralai/Mistral-7B-v0.1
    type: local

  - id: deepseek-ai/DeepSeek-R1
    type: remote
    base_url: http://other-server:8000

  - id: microsoft/phi-2
    type: remote
    base_url: http://phi-server:8000
~~~

If `models.yaml` is not provided, the server runs in **single-model mode** using
`MODEL_ID` from environment configuration.

The API can select models dynamically:

~~~json
{
  "prompt": "Explain transformers in one sentence",
  "model": "microsoft/phi-2"
}
~~~

Available models are discoverable via:

~~~bash
GET /v1/models
~~~

Response example:

~~~json
{
  "default_model": "mistralai/Mistral-7B-v0.1",
  "models": [
    "mistralai/Mistral-7B-v0.1",
    "deepseek-ai/DeepSeek-R1",
    "microsoft/phi-2"
  ]
}
~~~

---

## Architecture

At a high level:

Client  
→ **FastAPI Gateway** (Auth · Quotas · Routing · Limits · Logs · Metrics)  
→ **MultiModelManager**  
→ **ModelManager / HttpLLMClient**  
→ **Device (MPS / CPU / CUDA / Remote GPU)**  

Supporting components:

- **Database** (SQLite for dev/tests, Postgres in prod)
- **Prometheus** for metrics
- **Grafana** for visualizations
- **Docker / uv** for deployment & reproducibility

The system is a **multi-model routing platform**, not a single-model script.

---

## Component Responsibilities

### FastAPI app (`llm_server.main.create_app`)

- Composes middleware:
  - Logging
  - Limits
  - Metrics
  - CORS
- Initializes:
  - Redis (optional)
  - `MultiModelManager` or `ModelManager`
- Ensures the default model is loadable (`/readyz`)
- Registers routes:
  - `/healthz`
  - `/readyz`
  - `/v1/generate`
  - `/v1/stream`
  - `/v1/models`
  - `/metrics`

### API layer (`llm_server.api`)

#### `generate.py`

- `POST /v1/generate`
  - Validates `X-API-Key`
  - Resolves model via `MultiModelManager` (or single `ModelManager`)
  - Computes a `(model_id, prompt_hash, params_fingerprint)` cache key
  - Returns cached responses when available (`CompletionCache`)
  - Logs all requests to `InferenceLog`
- `POST /v1/stream`
  - Same validation and routing logic
  - Streams tokens via Server-Sent Events (`text/event-stream`)

#### `models.py` (API)

- `GET /v1/models`
  - Returns:
    - `default_model`
    - `models`: list of all available model IDs

#### `health.py`

- `GET /healthz` — simple liveness
- `GET /readyz` — DB + LLM readiness (calls `llm.ensure_loaded()`)

---

## Model Runtime (`llm_server.services`)

### `ModelManager` (Local backend)

- Lazy loads:
  - `AutoTokenizer.from_pretrained(model_id)`
  - `AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=..., device_map=None)`
- Moves model to:
  - `mps` (Apple Silicon) if available
  - Otherwise `cpu` (or explicit `model_device` from settings)
- Implements:
  - `.generate(prompt, max_new_tokens, temperature, top_p, top_k, stop, ...)`
  - `.stream(...)` using `TextIteratorStreamer`
- Applies default stop sequences to avoid the model continuing a “user turn”.

### `HttpLLMClient` (Remote backend)

- Forwards `.generate(...)` and `.stream(...)` to another LLM service over HTTP.
- Makes it easy to:
  - Split models across machines
  - Route to specialized backends (e.g. code model vs chat model)
  - Integrate hosted or custom runtimes.

### `MultiModelManager`

- Holds a mapping `{model_id: backend}` where backend is:
  - `ModelManager` (local) or
  - `HttpLLMClient` (remote) or
  - A test double.
- Only calls `ensure_loaded()` on the **default model** by default (for `/readyz`).
- Supports:
  - `default_id`
  - `.list_models()`
  - `__getitem__` / membership checks used by the routing layer.

---

## Persistence Layer (`llm_server.db`)

- `models.py`
  - `RoleTable` — user tiers / roles
  - `ApiKey` — API keys, roles, quotas, timestamps
  - `InferenceLog` — request/response payloads plus latency
  - `CompletionCache` — deduplication cache keyed by:
    - `model_id`
    - `prompt_hash`
    - `params_fingerprint`
- `session.py`
  - Async SQLAlchemy engine and `async_session_maker`
  - Defaults to SQLite (`sqlite+aiosqlite`) for dev/tests
  - Intended to be pointed at Postgres for production (`postgresql+asyncpg`)

---

## Evals Framework

There is an evaluation framework under `llm_server/eval/` that exercises the running server through its HTTP API.

Supported tasks include:

- **GSM8K** — grade-school math word problems
- **MMLU** — multi-task language understanding benchmark
- **MBPP** — code generation
- **Summarization**
- **Toxicity** (classification / scoring)

Example (GSM8K, 25 examples):

~~~bash
uv run eval \
  --task gsm8k \
  --max-examples 25 \
  --base-url http://localhost:8000 \
  --api-key <your-api-key>
~~~

These evals:

- Load datasets via `datasets.load_dataset`
- Build prompts
- Call your `/v1/generate` endpoint
- Compute task-specific metrics (accuracy, etc.)
- Print progress and final metrics

They turn the server into a benchmarkable system instead of a black box.

---

## Project Structure

~~~text
llm-server/
├── src/
│   └── llm_server/
│       ├── api/              # FastAPI routers: generate, health, models, etc.
│       ├── core/             # Config, logging, limits, metrics, errors, redis
│       ├── db/               # SQLAlchemy models + async session
│       ├── eval/             # Evaluation runners (GSM8K, MMLU, MBPP, etc.)
│       ├── services/         # ModelManager + MultiModelManager + HttpLLMClient
│       ├── providers/        # (Optional) pluggable providers / test doubles
│       └── main.py           # App factory (create_app) and lifespan wiring
│
├── migrations/               # Alembic migrations
├── scripts/                  # Admin and DB utilities (seed, list keys, etc.)
├── tests/                    # Pytest suite
├── models.yaml               # Optional multi-model routing config
├── Dockerfile.api            # Container image for the API service
├── docker-compose.yml        # Optional local stack (API + DB + Prometheus/Grafana)
├── pyproject.toml            # uv + project configuration
├── uv.lock                   # Locked dependency versions
└── README.md
~~~

---

## Core Capabilities

### 1. API Gateway Features

- API key validation via `X-API-Key`
- User tiers / roles (admin, default, free, etc.)
- Monthly quotas and usage tracking
- Request logging (method, path, latency, outcome)
- Rate limiting and concurrency patterns (middleware-based)
- Hooks for request tracing and observability
- Protection against abuse (throttling + quotas)

The goal is to look and feel like:

> **“An OpenAI-style backend for your own models.”**

---

### 2. Inference Layer

Runtime supports:

- Non-streaming token generation (`/v1/generate`)
- Streaming generation via SSE (`/v1/stream`)
- Standard text completion over prompts
- Parameters:
  - `max_new_tokens`
  - `temperature`
  - `top_p`
  - `top_k`
  - `stop` sequences
- Device selection:
  - Apple Silicon `mps`
  - `cpu`
  - (Future / external) `cuda` via settings

Works with any compatible Hugging Face causal LM model, such as:

- Mistral
- LLaMA
- Phi
- DeepSeek
- Custom fine-tuned models

---

## Observability

Built-in metrics include:

**General**
- Request counts (by route & status)
- Latency histograms
- Error counts

**LLM-specific**
- `llm_requests_total{route, model_id, cached}`
- `llm_tokens_total{direction, model_id}`  
- `llm_request_latency_ms{route, model_id}`

Exposed via:

- `/metrics` in Prometheus text format

Designed to plug directly into:

- Prometheus
- Grafana

Logs include:

- Method, path, status, latency
- Model ID and cache hits
- Token counts
- Exception messages for unhandled errors
- Per-request context (`request_id`, `api_key`, `client_ip`, etc.)

---

## Admin & Ops APIs

Administrative functionality is protected by admin-scoped API keys.

- `GET /v1/me/usage` – shows aggregated usage for the active API key
- `GET /v1/admin/keys` – list all API keys, roles, and timestamps
- `POST /v1/admin/keys` – create new API keys and assign roles
- `GET /v1/admin/logs` – query recent inference logs (paginated)

These endpoints convert the server from a simple model wrapper into a multi-tenant,
operator-friendly platform with real observability and governance.

---

## Testing

Tests use **Pytest** and focus on:

- API key validation
- Quota enforcement and rate limits
- Generate and stream endpoints
- Health and readiness probes
- Limits middleware behavior
- Multi-model routing logic (via test doubles)

To run tests:

~~~bash
uv run pytest
~~~

---

## Running Locally

Basic dev workflow (CPU or MPS):

1. Create and sync environment:

   ~~~bash
   uv sync --extra cpu
   ~~~

2. Run the API server:

   ~~~bash
   uv run serve
   ~~~

3. Hit the API:

   ~~~bash
   curl -X POST http://localhost:8000/v1/generate \
     -H "Content-Type: application/json" \
     -H "X-API-Key: <your-api-key>" \
     -d '{"prompt": "Hello, world!", "max_new_tokens": 32}'
   ~~~

4. Check health and models:

   ~~~bash
   curl http://localhost:8000/healthz
   curl http://localhost:8000/readyz
   curl http://localhost:8000/v1/models
   curl http://localhost:8000/metrics
   ~~~

---

## Notes

- In development and tests, SQLite is used for simplicity.
- The design is ready to switch to Postgres by changing `DATABASE_URL`.
- The LLM backend supports:
  - Single-model mode (`ModelManager`)
  - Multi-model mode (`MultiModelManager` + optional remote models)
- The eval framework makes it easy to track performance across models.

This repository is meant to be a realistic, end-to-end example of how to host, secure, and observe LLM inference in a way that is legible to production engineering teams.