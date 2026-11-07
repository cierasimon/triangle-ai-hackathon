"""FastAPI app: /api routers, the share page, and the built frontend at /."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api import api_router, public_router
from app.config import FRONTEND_DIST

app = FastAPI(title="PaperPhysics")
app.include_router(api_router)
app.include_router(public_router)


@app.get("/api/health")
def health():
    return {"ok": True}


if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/", include_in_schema=False)
def index():
    page = FRONTEND_DIST / "index.html"
    if page.is_file():
        return FileResponse(page)
    return HTMLResponse(
        "<p>Frontend not built. Use <code>make dev</code> and open "
        '<a href="http://localhost:5173">http://localhost:5173</a>, '
        "or run <code>make build</code>.</p>"
    )


@app.get("/{name}", include_in_schema=False)
def root_file(name: str):
    """Files Vite copies to the build root (favicon etc.), never outside frontend/dist."""
    path = (FRONTEND_DIST / name).resolve()
    if path.parent == FRONTEND_DIST.resolve() and path.is_file():
        return FileResponse(path)
    raise HTTPException(404, "not found")
