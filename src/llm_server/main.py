# src/llm_server/main.py
from __future__ import annotations

import orjson
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llm_server.core.config import settings
from llm_server.core import logging as logging_config
from llm_server.core import metrics, limits
from llm_server.core.redis import init_redis, close_redis
from llm_server.services.llm import build_llm_from_settings  # <--- new import


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------- startup ----------
    import logging

    logging.getLogger("uvicorn.error").info(
        "CORS allow_origins=%s | env=%s | debug=%s | redis_enabled=%s",
        settings.cors_allowed_origins,
        settings.env,
        settings.debug,
        settings.redis_enabled,
    )

    # init redis
    try:
        app.state.redis = await init_redis()
    except Exception as e:
        logging.getLogger("uvicorn.error").exception("Redis init failed: %s", e)
        app.state.redis = None

    # init LLM (single or multi-model, depending on config / models.yaml)
    try:
        llm = build_llm_from_settings()
        # Make it visible to dependencies (generate.py, health.py, etc.)
        app.state.llm = llm

        # For readiness: ensure the default/local model can be loaded
        if hasattr(llm, "ensure_loaded"):
            llm.ensure_loaded()
    except Exception as e:
        logging.getLogger("uvicorn.error").exception("LLM init failed: %s", e)
        app.state.llm = None

    yield

    # ---------- shutdown ----------
    await close_redis(getattr(app.state, "redis", None))


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.service_name,
        description="Backend service for running LLM inference",
        version=settings.version,
        debug=settings.debug,
        lifespan=lifespan,
        json_dumps=lambda v, *, default: orjson.dumps(v, default=default).decode(),
        json_loads=orjson.loads,
    )

    # --- Middleware ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logging_config.setup(app)
    limits.setup(app)
    metrics.setup(app)

    # --- Routers ---
    from llm_server.api import health, generate, models, admin

    app.include_router(health.router)
    app.include_router(generate.router)
    app.include_router(models.router)
    app.include_router(admin.router)

    return app


app = create_app()