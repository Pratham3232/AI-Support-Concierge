import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import routes_chat, routes_sessions, routes_traces
from app.api.errors import HelixError, helix_error_handler
from app.db.session import engine, init_db
from app.obs.logging import configure_logging
from app.settings import settings

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    # ADK reads GOOGLE_API_KEY from os.environ directly. Pydantic-settings only
    # loads it into `settings`, so we bridge the gap here.
    if settings.google_api_key:
        os.environ.setdefault("GOOGLE_API_KEY", settings.google_api_key)
    await init_db()
    try:
        yield
    finally:
        await engine.dispose()


async def _catch_all_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled_exception", exc_type=type(exc).__name__, detail=str(exc)[:200])
    return JSONResponse(
        status_code=500,
        content={
            "type": "https://docs.helix.example/errors/internal_error",
            "title": "INTERNAL_ERROR",
            "status": 500,
            "detail": f"{type(exc).__name__}: {str(exc)[:200]}",
        },
    )


app = FastAPI(title="Helix SROP", version="0.1.0", lifespan=lifespan)
app.add_exception_handler(HelixError, helix_error_handler)
app.add_exception_handler(Exception, _catch_all_handler)

app.include_router(routes_sessions.router, prefix="/v1")
app.include_router(routes_chat.router, prefix="/v1")
app.include_router(routes_traces.router, prefix="/v1")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}
