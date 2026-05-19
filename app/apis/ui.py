from __future__ import annotations

import json

from fastapi import HTTPException, APIRouter
from starlette.responses import HTMLResponse, FileResponse, JSONResponse

from app.kggen_logger import kggen_logger
from app.utils import TEMPLATE_PATH, STATIC_DIR, refresh_examples, DATA_ROOT
from kg_gen.models import Graph
from kg_gen.utils.visualize_kg import _build_view_model


ui_router = APIRouter(prefix="/ui", tags=["User Interface"])


@ui_router.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="index.html missing")
    kggen_logger.debug("Serving index page")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@ui_router.get("/template")
async def serve_template() -> FileResponse:
    kggen_logger.debug("Serving visualization template from %s", TEMPLATE_PATH)
    return FileResponse(TEMPLATE_PATH, media_type="text/html")


@ui_router.get("/examples")
async def list_examples() -> JSONResponse:
    EXAMPLE_INDEX = refresh_examples()
    kggen_logger.debug(f"Listing built-in example graphs. {EXAMPLE_INDEX = }")
    items = [
        {"slug": example.slug, "title": example.title, "wiki_url": example.wiki_url}
        for example in sorted(
            EXAMPLE_INDEX.values(), key=lambda item: item.title.lower()
        )
    ]
    return JSONResponse(items)


@ui_router.get("/examples/{slug}")
async def load_example(slug: str) -> JSONResponse:
    EXAMPLE_INDEX = refresh_examples()
    example = EXAMPLE_INDEX.get(slug)
    if example is None:
        raise HTTPException(status_code=404, detail=f"Example '{slug}' not found")

    if not example.path.exists():
        kggen_logger.error("Example graph missing: %s", example.path)
        raise HTTPException(status_code=404, detail=f"Example '{slug}' is unavailable")

    try:
        payload = json.loads(example.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        kggen_logger.exception("Failed to parse example graph %s", slug)
        raise HTTPException(
            status_code=500, detail=f"Example '{slug}' is invalid: {exc}"
        )

    kggen_logger.info("Loaded example graph '%s' from %s", slug, example.path)
    return JSONResponse(payload)


@ui_router.post("/graph/view")
async def build_view(graph: Graph) -> JSONResponse:
    """Convert a raw KGGen graph payload into the template view model."""
    try:
        view = _build_view_model(graph)
    except Exception as exc:  # pragma: no cover - defensive
        kggen_logger.exception("Failed to build view model")
        raise HTTPException(status_code=500, detail=f"Failed to build view: {exc}")
    kggen_logger.info(
        "View model ready: nodes=%s edges=%s", len(view["nodes"]), len(view["edges"])
    )
    return JSONResponse({"view": view, "graph": graph.model_dump(mode="json")})


@ui_router.post("/add_graph")
async def add_graph(graph: Graph, title: str) -> str:
    """
    Paste your generated graph here to view it in the UI later (load via existing graphs).
    """
    try:
        view = _build_view_model(graph)
    except Exception as exc:  # pragma: no cover - defensive
        kggen_logger.exception("Failed to build view model")
        raise HTTPException(status_code=400, detail=f"Failed to build view: {exc}")
    kggen_logger.info(
        "View model ready: nodes=%s edges=%s", len(view["nodes"]), len(view["edges"])
    )
    out_file_path = str(DATA_ROOT / f"{title}.json")
    try:
        graph.to_file(file_path=out_file_path)
    except Exception as exc:
        kggen_logger.exception("Failed to save new graph")
        raise HTTPException(status_code=500, detail=f"Failed to save graph: {exc}")
    return out_file_path
