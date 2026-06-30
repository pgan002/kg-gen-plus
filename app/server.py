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


@asynccontextmanager
async def lifespan(_: FastAPI):
    await config_logger()
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


# Serve static files (CSS, JS, etc.) - must be mounted after all routes
app.mount("/ui", StaticFiles(directory=APP_DIR / "static", html=True), name="static")

dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
