"""POST /api/export/{run_id}, share files, and the mobile page GET /s/{share_id}.

Step 0: the GIF is a single placeholder frame; export/gif.py (P4.1) renders the real one.
"""

import html
import io

import qrcode
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, Response
from PIL import Image, ImageDraw

from app import store
from app.config import get_settings
from app.contracts import ExportResponse

router = APIRouter()
public_router = APIRouter()  # mounted at the root, not under /api


def _placeholder_gif(text: str) -> bytes:
    image = Image.new("RGB", (480, 270), (250, 250, 247))
    ImageDraw.Draw(image).text((20, 125), text, fill=(30, 30, 30))
    buf = io.BytesIO()
    image.save(buf, format="GIF")
    return buf.getvalue()


@router.post("/export/{run_id}", response_model=ExportResponse)
def export(run_id: str, session: store.Session = Depends(store.session_dep)):
    run = session.get_run(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    share_id = store.new_id()
    store.put_share(store.Share(share_id=share_id, run_id=run_id,
                                gif=_placeholder_gif(f"FAKE GIF for Run {run.index}")))
    session.log("export", run_id=run_id, share_id=share_id)
    return ExportResponse(
        gif_url=f"/api/share/{share_id}/run.gif",
        share_url=f"{get_settings().public_base_url}/s/{share_id}",
        qr_png_url=f"/api/share/{share_id}/qr.png",
    )


@router.get("/share/{share_id}/run.gif")
def share_gif(share_id: str):
    share = store.get_share(share_id)
    if not share:
        raise HTTPException(404, "not found")
    return Response(share.gif, media_type="image/gif")


@router.get("/share/{share_id}/qr.png")
def share_qr(share_id: str):
    if not store.get_share(share_id):
        raise HTTPException(404, "not found")
    buf = io.BytesIO()
    qrcode.make(f"{get_settings().public_base_url}/s/{share_id}").save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@public_router.get("/s/{share_id}", response_class=HTMLResponse)
def share_page(share_id: str):
    if not store.get_share(share_id):
        raise HTTPException(404, "not found")
    gif = f"/api/share/{html.escape(share_id)}/run.gif"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>PaperPhysics</title>
<style>body{{font-family:system-ui;margin:16px;text-align:center}}img{{max-width:100%}}
a{{display:inline-block;margin-top:16px;padding:12px 20px;background:#111;color:#fff;
border-radius:8px;text-decoration:none}}</style></head>
<body><h1>PaperPhysics</h1><img src="{gif}" alt="simulation"><br>
<a href="{gif}" download="paperphysics.gif">Download GIF</a></body></html>"""
