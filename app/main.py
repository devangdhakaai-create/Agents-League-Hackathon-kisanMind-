# =============================================================================
# app/main.py
# =============================================================================
# PURPOSE: FastAPI application entry point.
# Creates the app, registers middleware, mounts the router, runs startup
# checks. This is the file uvicorn loads when the server starts.
# =============================================================================

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.db.database import create_tables, verify_connection
from app.db import models  # noqa: F401 — import triggers model registration with Base
from app.api.routes import router

# =============================================================================
# LOGGING SETUP
# =============================================================================
# Configure before anything else so all module-level loggers are captured.
# In development: DEBUG level, human-readable format.
# In production: INFO level, structured format for Azure Monitor ingestion.
# =============================================================================

log_level = logging.DEBUG if settings.ENVIRONMENT == "development" else logging.INFO

logging.basicConfig(
    level=log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,  # Azure Container Apps captures stdout as logs
)

logger = logging.getLogger(__name__)


# =============================================================================
# LIFESPAN — STARTUP AND SHUTDOWN
# =============================================================================
# FastAPI's lifespan context manager replaces the deprecated @app.on_event.
# Code before `yield` runs at startup. Code after `yield` runs at shutdown.
# If startup raises, the server refuses to start — loud failure by design.
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: verify database, create tables, log readiness.
    Shutdown: log clean exit (connection pool closes automatically).
    """
    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"LLM model: {settings.LLM_MODEL}")
    logger.info(f"LLM endpoint: {settings.GITHUB_MODELS_BASE_URL}")
    logger.info("=" * 60)

    # Verify PostgreSQL is reachable — crash loudly if not
    # Better to fail here than silently during the first request
    logger.info("Verifying database connection...")
    try:
        verify_connection()
        logger.info("Database connection: OK")
    except Exception as e:
        logger.critical(f"Database connection failed at startup: {e}")
        logger.critical("Refusing to start — fix DATABASE_URL in .env")
        sys.exit(1)  # non-zero exit code triggers Azure Container App restart

    # Create all tables if they don't exist
    # models must be imported above before this runs
    logger.info("Running database table creation...")
    create_tables()
    logger.info("Database tables: OK")

    # Log all registered routes for debugging
    logger.info("Registered API routes:")
    for route in app.routes:
        if hasattr(route, "methods"):
            methods = ", ".join(route.methods)
            logger.info(f"  {methods:20s} {route.path}")

    logger.info(f"{settings.APP_NAME} startup complete — ready to serve requests")

    yield  # server runs here — everything below runs at shutdown

    logger.info(f"{settings.APP_NAME} shutting down cleanly")


# =============================================================================
# FASTAPI APP INSTANCE
# =============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    # OpenAPI docs available at /docs (Swagger) and /redoc
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# =============================================================================
# CORS MIDDLEWARE
# =============================================================================
# Required for the frontend (HTML form) to call the API from a browser.
# In development: allow all origins (*).
# In production: restrict to your Azure Static Web App URL.
# Controlled by CORS_ORIGINS in .env — no code change needed for deployment.
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,   # ["*"] in dev, specific URL in prod
    allow_credentials=True,
    allow_methods=["GET", "POST"],         # only what KisanMind needs
    allow_headers=["*"],
)


# =============================================================================
# GLOBAL EXCEPTION HANDLER
# =============================================================================
# Catches any unhandled exception that escapes route handlers.
# Returns a consistent JSON error shape instead of FastAPI's default
# HTML error page — critical for the frontend to handle errors gracefully.
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches all unhandled exceptions at the application level.
    Logs the full traceback for debugging while returning a clean
    JSON response to the client.
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": (
                "An unexpected error occurred. "
                "Please retry or contact support."
                if settings.ENVIRONMENT == "production"
                else str(exc)  # expose error details in development only
            ),
        },
    )


# =============================================================================
# ROUTER REGISTRATION
# =============================================================================
# All routes from routes.py are mounted under /api/v1.
# Versioned prefix means future breaking changes go to /api/v2
# without removing /api/v1 — zero downtime upgrades.
# =============================================================================

app.include_router(
    router,
    prefix="/api/v1",       # all routes become /api/v1/advisory, etc.
    tags=["KisanMind API"], # groups routes in /docs
)


# =============================================================================
# ROOT REDIRECT
# =============================================================================

@app.get("/", include_in_schema=False)
def root():
    """
    Root endpoint — returns basic API info.
    Not included in OpenAPI schema (include_in_schema=False).
    In production, the frontend is served separately (static files or
    Azure Static Web Apps) — this just confirms the API is alive.
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# =============================================================================
# UVICORN ENTRYPOINT
# =============================================================================
# Allows running directly with: python app/main.py
# In production, use: uvicorn app.main:app --host 0.0.0.0 --port 8000
# In Docker: CMD is set in Dockerfile — this block is not executed.
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",        # listen on all interfaces
        port=8000,
        reload=True,           # hot reload in development — disable in prod
        log_level="debug" if settings.ENVIRONMENT == "development" else "info",
    )