"""FastAPI app: /api routers, the share page, and the built frontend at /."""

from fastapi import FastAPI
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
