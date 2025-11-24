# LLM Server  
### A Production-Style API Gateway + Inference Runtime for Large Language Models

This project is a self-hosted, production-inspired **LLM serving platform** built with FastAPI, Hugging Face Transformers, and PyTorch. It is designed to mirror real-world architecture patterns used by companies deploying foundation models behind internal and external APIs.

Rather than being just a demo model runner, this repository focuses on:

- Infrastructure
- Security
- Quotas
- Observability
- Scalability patterns
- Clean separation of concerns

It is designed to be a **portfolio-grade system-level project** demonstrating my ability to design and build ML/AI infrastructure.

---

## Key Features

- FastAPI-based LLM gateway
- API key authentication
- Rate limiting + concurrency limits
- Usage quotas (monthly limits)
- Prometheus metrics + Grafana dashboards
- SQLAlchemy + Alembic for persistence
- Streaming & non-streaming generation
- Hugging Face Transformers backend
- Docker support
- Test suite with Pytest
- Configurable for CPU, Apple Silicon (MPS), or CUDA

The architecture cleanly separates:

- Gateway API — validates requests, manages users/quotas, logs history
- LLM Runtime — loads and serves the actual model in-process
- Database layer — tracks users, API keys, quotas and request history
- Metrics layer — exposes structured telemetry

This design follows real patterns used in production AI platforms.

---

## Architecture

At a high level:

Client  
→ FastAPI Gateway (Auth · Quotas · Limits · Logging · Metrics)  
→ Model Runtime (`ModelManager` with Transformers + PyTorch)  
→ Device (MPS / CPU / CUDA)  

Supporting components:

- **Database** (SQLite for dev/tests, Postgres in production) for:
  - API keys and roles
  - Quotas and usage
  - Inference logs
  - Completion cache
- **Prometheus** for metrics collection
- **Grafana** for visualization
- **Docker / uv** for packaging, reproducible environments, and deployment

### Component Responsibilities

- **FastAPI app (`llm_server.main.create_app`)**
  - Composes:
    - Middlewares (logging, limits, metrics, CORS)
    - Routers (`/health`, `/readyz`, `/v1/generate`, `/v1/stream`, `/metrics`)
  - Manages lifespan:
    - On startup:
      - Instantiates a `ModelManager` and stores it in `app.state.llm`
      - Ensures DB connectivity (indirectly, via first use)
    - On shutdown:
      - Cleans up state if needed

- **API layer (`llm_server.api`)**
  - `generate.py`
    - `POST /v1/generate`
      - Validates API key (`X-API-Key`)
      - Checks quotas and role
      - Computes prompt and parameter fingerprints
      - Uses `CompletionCache` for deduplication
      - Calls `llm.generate(...)` and logs to `InferenceLog`
    - `POST /v1/stream`
      - Same validation path, but uses `llm.stream(...)`
      - Returns Server-Sent Events (`text/event-stream`)
  - `health.py`
    - `GET /health` — liveness probe
    - `GET /readyz` — readiness probe (calls `llm.ensure_loaded()`)

- **Core platform (`llm_server.core`)**
  - `config.py` — Pydantic settings loaded from environment variables
  - `logging.py` — structured access logging and error logging
  - `limits.py` — concurrency / rate-limiting hooks
  - `metrics.py` — Prometheus instrumentation for requests and latency
  - `errors.py`, `redis.py` — error handling and optional Redis plumbing

- **Model runtime (`llm_server.services.llm.ModelManager`)**
  - Lazy-loads:
    - `AutoTokenizer.from_pretrained(settings.model_id)`
    - `AutoModelForCausalLM.from_pretrained(settings.model_id, torch_dtype=..., device_map=None)`
  - Moves the model to the selected device:
    - `mps` on Apple Silicon if available, otherwise `cpu` (and future CUDA support via extras)
  - Implements:
    - `generate(...)` — non-streaming completion, returns a single string
    - `stream(...)` — streaming generation using `TextIteratorStreamer`
  - Applies stop sequences and safe defaults (temperature, top-p, top-k, etc.)

- **Persistence layer (`llm_server.db`)**
  - `models.py`
    - `RoleTable` — user tiers / roles
    - `ApiKey` — API keys, role association, quotas, created_at
    - `InferenceLog` — full request/response and latency logging
    - `CompletionCache` — deduplication cache keyed by `(model_id, prompt_hash, params_fingerprint)`
  - `session.py`
    - Async SQLAlchemy engine and `async_session_maker`
    - Uses SQLite (`sqlite+aiosqlite`) in dev/test, designed to support Postgres (`asyncpg`) in production

- **Providers (`llm_server.providers`)**
  - Abstraction layer for external or remote LLM backends.
  - Tests monkeypatch this layer to inject a `DummyModelManager` instead of loading a real model.
  - In a future multi-service setup, this is where HTTP-based LLM clients would live.

- **Admin scripts (`scripts/`)**
  - CLI utilities for:
    - Seeding API keys
    - Listing keys and roles
    - Inspecting the DB

### Request Flow (Generate)

1. Client sends `POST /v1/generate` with JSON body and header `X-API-Key: <key>`.
2. Middlewares:
   - Logging middleware records incoming request.
   - Limits middleware enforces concurrency.
   - Metrics middleware starts latency timer and increments counters.
3. FastAPI resolves dependencies:
   - `get_api_key` checks DB for the key, role, and quotas.
   - `get_llm` returns `app.state.llm` (a `ModelManager` instance).
   - Request body is validated into `GenerateRequest`.
4. Handler:
   - Hashes prompt and generation parameters.
   - Checks `CompletionCache`:
     - If hit: returns cached output and logs `InferenceLog`.
     - If miss: calls `llm.generate(...)`.
5. `ModelManager`:
   - Ensures model/tokenizer are loaded.
   - Runs `model.generate(...)` on the chosen device.
   - Applies stop sequences and returns output text.
6. Handler:
   - Saves `CompletionCache` and `InferenceLog`.
   - Returns `{ "model": ..., "output": ..., "cached": false }`.
7. Middleware unwinds:
   - Metrics middleware records final status/latency.
   - Logging middleware writes a structured access log line.

This architecture is intentionally closer to a real service like an “OpenAI-style backend for your own models” than to a simple notebook-based demo.

---

## Project Structure

    llm-server/
    ├── src/
    │   └── llm_server/
    │       ├── api/              # FastAPI routers: generate, health, etc.
    │       ├── core/             # Config, logging, limits, metrics, errors
    │       ├── db/               # SQLAlchemy models + async session
    │       ├── services/         # ModelManager (Transformers + PyTorch)
    │       ├── providers/        # Pluggable LLM clients (for tests/remote runtimes)
    │       └── main.py           # App factory (create_app) and lifespan wiring
    │
    ├── migrations/               # Alembic migrations
    ├── scripts/                  # Admin and DB utilities
    ├── tests/                    # Pytest suite (auth, quotas, limits, generate, health)
    ├── Dockerfile.api            # Container image for the API service
    ├── docker-compose.yml        # Optional local stack (API + DB + Prometheus/Grafana)
    ├── pyproject.toml            # uv + setuptools project config
    ├── uv.lock                   # Locked dependency versions
    └── README.md

---

## Core Capabilities

### 1. API Gateway Features

- API key validation via `X-API-Key`
- User tiers / roles (via `RoleTable`)
- Monthly quotas and usage tracking
- Request logging (method, path, latency, outcome)
- Rate limiting and concurrency patterns (via middleware hooks)
- Request tracing hooks and observability
- Protection against abuse (throttling + quota enforcement)

The goal is to look and feel like:

**“An OpenAI-style backend for your own models”**

rather than a traditional ML demo script.

---

### 2. Inference Layer

The runtime supports:

- Non-streaming token generation (`/v1/generate`)
- Streaming generation via Server-Sent Events (`/v1/stream`)
- Standard completion over prompts (chat-style prompting is handled at the prompt level)
- Model configuration via parameters:
  - `max_new_tokens`
  - `temperature`
  - `top_p`
  - `top_k`
  - `stop` sequences
- Device selection:
  - Apple Silicon (MPS)
  - CPU
  - (Future) CUDA 12.1 via extras

Designed to work with Hugging Face Causal LM models such as:

- Mistral
- LLaMA
- Phi
- DeepSeek
- Custom HF models you host yourself

---

### 3. Observability

Built-in metrics include:

- Total requests (by route and status)
- Latency histograms (e.g., p95-ready)
- Error counts (by type)
- Potential per-key or per-model metrics

These metrics are exported via:

- `/metrics` (Prometheus text format)

and can be visualized with:

- Prometheus + Grafana dashboards.

Structured logs include:

- Request ID (if configured)
- Method, path, status, latency
- Exception messages for unhandled errors

---

## Testing

Tests are written using **Pytest** and focus on:

- API key validation (auth, 401/403 paths)
- Quota enforcement and rate limits
- Generate and stream endpoints
- Health and readiness endpoints
- Limits middleware behavior

The test suite uses:

- `pytest-anyio` for async tests
- `asgi-lifespan` to ensure startup/shutdown are invoked
- `httpx.AsyncClient` + `ASGITransport` to call the app in-process
- A dummy `ModelManager` injected via monkeypatching so tests never hit Hugging Face or load real models

To run tests:

    uv run pytest

---

## Running Locally

Basic dev workflow (CPU or MPS):

1. Create and sync the environment (example with CPU extras):

    uv sync --extra cpu

2. Run the API server:

    uv run serve

3. Hit the API (example):

    curl -X POST http://localhost:8000/v1/generate \
      -H "Content-Type: application/json" \
      -H "X-API-Key: <your-api-key>" \
      -d '{"prompt": "Hello, world!", "max_new_tokens": 32}'

4. Check health and metrics:

    curl http://localhost:8000/health
    curl http://localhost:8000/readyz
    curl http://localhost:8000/metrics

---

## Notes

- In development and tests, SQLite is used for simplicity.
- The design is ready to switch to Postgres for production by changing `DATABASE_URL`.
- The LLM backend currently uses a single in-process `ModelManager`, but the provider layer makes it straightforward to:
  - Call a remote LLM service
  - Route between multiple models
  - Evolve into a multi-service architecture.

This repository is meant to be a realistic, end-to-end example of how to host, secure, and observe LLM inference in a way that is legible to production engineering teams.