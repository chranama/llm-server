# src/llm_server/cli.py
import os
import uvicorn

def serve():
    """
    Entry point for `uv run serve`.
    - ENV=dev or DEV=1 enables auto-reload (single worker).
    - HOST/PORT/WORKERS env vars override defaults.
    """
    env = os.getenv("ENV", "").lower()
    dev_mode = env == "dev" or os.getenv("DEV") == "1"

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    # In dev we force single worker + reload for hot-reload
    if dev_mode:
        uvicorn.run(
            "llm_server.main:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
            proxy_headers=True,
        )
        return

    # Production-ish defaults
    workers = int(os.getenv("WORKERS", "1"))
    uvicorn.run(
        "llm_server.main:create_app",
        factory=True,
        host=host,
        port=port,
        workers=workers,
        proxy_headers=True,
    )