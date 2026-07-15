from __future__ import annotations

import dspy
import toml
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from starlette.responses import RedirectResponse

import logging
import logging.config

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.apis.kg_construct import kgc_router
from app.apis.ui import ui_router
from app.schemas import HeartBeatResponse
from app.utils import APP_DIR


async def config_logger():
    with open(Path(__file__).parent / "logging.yaml") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)


description_path = Path(__file__).parent / "api_doku.md"
api_description = description_path.read_text()

logger = logging.getLogger(__name__)


def _load_mcp_asgi_app():
    """Load the kg-gen MCP server and return its ASGI app for in-process mounting.

    The MCP server lives at ``mcp/server.py`` (kept non-importable-as-package to
    avoid shadowing the installed ``mcp`` SDK), so it is loaded by file path.
    Returns ``None`` if the optional ``fastmcp`` dependency is not installed, so
    the HTTP API still starts without it.
    """
    import importlib.util

    mcp_server_path = Path(__file__).parent.parent / "mcp" / "server.py"
    spec = importlib.util.spec_from_file_location("kg_mcp_server", mcp_server_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # path="/" so the endpoint is exactly the mount point ("/mcp").
    return module.mcp.http_app(path="/")


try:
    mcp_asgi_app = _load_mcp_asgi_app()
except Exception as exc:  # pragma: no cover - depends on optional extra
    mcp_asgi_app = None
    logger.warning("MCP server not mounted (install 'kg-gen[mcp]'): %s", exc)


@asynccontextmanager
async def lifespan(app_: FastAPI):
    await config_logger()
    # The mounted MCP app has its own lifespan (session manager); run it nested
    # so it starts/stops together with the FastAPI app.
    if mcp_asgi_app is not None:
        async with mcp_asgi_app.lifespan(app_):
            yield
    else:
        yield


def get_version() -> str:
    path = Path(__file__).parent.parent / "pyproject.toml"
    with open(str(path)) as pyproject_toml_file:
        pyproject = toml.loads(pyproject_toml_file.read())
        version: str = pyproject["project"]["version"]
        return version


app = FastAPI(
    title="KGGen+",
    version=get_version(),
    description=api_description,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/heartbeat", name="heartbeat")
def heartbeat() -> HeartBeatResponse:
    return HeartBeatResponse(is_alive=True)


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url=app.docs_url or "/docs")


app.include_router(kgc_router)
app.include_router(ui_router)

# Mount the kg-gen MCP server (streamable HTTP) in-process, so it starts and
# stops together with the FastAPI app. Available at /mcp when fastmcp is installed.
if mcp_asgi_app is not None:
    app.mount("/mcp", mcp_asgi_app)


# Serve static files (CSS, JS, etc.) - must be mounted after all routes
app.mount("/ui", StaticFiles(directory=APP_DIR / "static", html=True), name="static")

dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
