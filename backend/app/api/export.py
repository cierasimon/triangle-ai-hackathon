"""Take it home (P4.3): POST /api/export/{run_id} renders the animation, stores it under a
random share id, and returns its URL, a QR code for the phone, and the share page URL.

The share page GET /s/{share_id} is mounted at the root (not under /api) so the QR link is
short. Its base URL is PUBLIC_BASE_URL; while that is still the localhost default, the
address the app was opened on is used instead, so a phone on the same Wi-Fi (or a tunnel)
gets a link that works. Opened on localhost itself, the link uses this computer's Wi-Fi
address (same port) so a phone on the same network can still scan it.
"""

import html
import io
import logging
import socket

import qrcode
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from app import store
from app.config import get_settings
from app.contracts import ExportRequest, ExportResponse
from app.export.gif import card_lines, render

router = APIRouter()
public_router = APIRouter()  # mounted at the root, not under /api
log = logging.getLogger(__name__)

LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")
EXT = {"image/gif": "gif", "image/png": "png"}


def lan_ip() -> str | None:
    """This computer's address on the local network (no packet is sent), or None."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
    except OSError:
        return None
    return None if ip.startswith("127.") else ip


def public_base(request: Request) -> str:
    configured = get_settings().public_base_url
    if not any(h in configured for h in LOCAL_HOSTS):
        return configured
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    if host and not any(host.startswith(h) for h in LOCAL_HOSTS):
        return f"{proto}://{host}"
    ip = lan_ip()
    if host and ip:  # opened on localhost: same port, but on the Wi-Fi address
        port = host.rsplit(":", 1)[1] if host.count(":") == 1 else ""
        return f"{proto}://{ip}{':' + port if port else ''}"
    return configured


@router.post("/export/{run_id}", response_model=ExportResponse)
def export(run_id: str, request: Request, body: ExportRequest | None = Body(default=None),
           session: store.Session = Depends(store.session_dep)):
    run = session.get_run(run_id)
    if run is None and body and body.run and body.run.run_id == run_id:
        run = body.run  # the server restarted since: render the run the page sent
    if run is None:
        raise HTTPException(404, "unknown run — run the simulation again, then export")
    rendered = render(run)
    share_id = store.new_id()
    store.put_share(store.Share(share_id=share_id, data=rendered.data,
                                media_type=rendered.media_type, run_id=run_id,
                                title="latest run", lines=card_lines(run)[1:]))
    session.log("export", run_id=run_id, share_id=share_id, media_type=rendered.media_type,
                bytes=len(rendered.data), render_s=round(rendered.render_s, 3))
    ext = EXT[rendered.media_type]
    return ExportResponse(
        gif_url=f"/api/share/{share_id}/run.{ext}",
        share_url=f"{public_base(request)}/s/{share_id}",
        qr_png_url=f"/api/share/{share_id}/qr.png",
    )


def _share(share_id: str) -> store.Share:
    share = store.get_share(share_id)
    if not share:
        raise HTTPException(404, "not found — exports live until the server restarts")
    return share


@router.get("/share/{share_id}/run.{ext}")
def share_file(share_id: str, ext: str):
    share = _share(share_id)
    return Response(share.data, media_type=share.media_type,
                    headers={"Cache-Control": "public, max-age=86400"})


@router.get("/share/{share_id}/download")
def share_download(share_id: str):
    share = _share(share_id)
    name = f"paperphysics-{share.title.lower().replace(' ', '-')}.{EXT[share.media_type]}"
    return Response(share.data, media_type=share.media_type,
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/share/{share_id}/qr.png")
def share_qr(share_id: str, request: Request, download: bool = False):
    share = _share(share_id)
    img = qrcode.make(f"{public_base(request)}/s/{share_id}", border=2, box_size=8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    headers = {}
    if download:
        name = f"paperphysics-{share.title.lower().replace(' ', '-')}-qr.png"
        headers["Content-Disposition"] = f'attachment; filename="{name}"'
    return Response(buf.getvalue(), media_type="image/png", headers=headers)


@public_router.get("/s/{share_id}", response_class=HTMLResponse)
def share_page(share_id: str):
    share = _share(share_id)
    sid = html.escape(share_id)
    ext = EXT[share.media_type]
    lines = "".join(f"<p>{html.escape(line)}</p>" for line in share.lines)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PaperPhysics · {html.escape(share.title)}</title>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:16px;background:#0b0b0c;
color:#ececec;text-align:center}}
main{{max-width:960px;margin:0 auto}}
h1{{font-size:1.4rem;margin:8px 0 12px}}
img{{width:100%;height:auto;border-radius:12px;border:1px solid #2a2a2d;background:#141415}}
p{{margin:6px 0;color:#8c8c8c;font-size:.95rem}}
a.btn{{display:block;margin:16px auto 8px;max-width:360px;padding:14px 20px;background:#ececec;
color:#0b0b0c;border-radius:10px;text-decoration:none;font-weight:600}}
small{{color:#8c8c8c}}
</style></head>
<body><main>
<h1>PaperPhysics · {html.escape(share.title)}</h1>
<img src="/api/share/{sid}/run.{ext}" alt="simulation of the drawn ramp">
{lines}
<a class="btn" href="/api/share/{sid}/download" download>Download {ext.upper()}</a>
<small>On iPhone you can also press and hold the image, then “Save to Photos”.</small>
</main></body></html>"""
