"""POST /upload, POST /corners, GET /files/...: photo → capture → recognition → overlay."""

import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import Response

from app import store
from app.contracts import CornersRequest, Recognition, UploadResponse
from app.cv import capture as cap
from app.cv.recognize import overlay_png, recognize

router = APIRouter()
log = logging.getLogger(__name__)


def _recognize(stored: store.StoredImage, result: cap.Capture) -> Recognition:
    """Run recognition and keep the overlay; never fails the request on a CV bug."""
    try:
        res = recognize(result, stored.image_id)
        rec, image = res.recognition, res.image
    except Exception:  # keep the demo alive: report it as a warning instead
        log.exception("recognition failed for %s", stored.image_id)
        h, w = result.image.shape[:2]
        rec = Recognition(image_id=stored.image_id, paper_found=result.paper_found,
                          paper_corners=result.paper_corners, warped_size=(w, h), lines=[],
                          ball=None, warnings=[*result.warnings, "recognition failed — try "
                                               "another photo or tap the paper corners"])
        image = result.image
    stored.recognition = rec
    stored.overlay_png = overlay_png(image, rec)
    return rec


@router.post("/upload", response_model=UploadResponse)
async def upload(photo: UploadFile, session: store.Session = Depends(store.session_dep)):
    data = await photo.read()
    try:
        image = cap.load_image(data)
    except Exception as exc:  # Pillow raises many types for bad files
        raise HTTPException(400, f"not an image: {exc}") from exc
    image_id = store.new_id("img_")
    stored = store.StoredImage(image_id=image_id,
                               content_type=photo.content_type or "application/octet-stream",
                               original=data)
    session.images[image_id] = stored
    rec = _recognize(stored, cap.capture(image))
    session.log("upload", image_id=image_id, filename=photo.filename, warnings=rec.warnings)
    return UploadResponse(
        image_id=image_id,
        recognition=rec,
        overlay_png_url=_overlay_url(session, image_id),
    )


@router.post("/corners", response_model=Recognition)
def corners(body: CornersRequest, session: store.Session = Depends(store.session_dep)):
    stored = session.images.get(body.image_id)
    if not stored:
        raise HTTPException(404, "unknown image_id")
    if len(body.corners) != 4:
        raise HTTPException(422, "need exactly 4 corners")
    image = cap.load_image(stored.original)
    stored.overlay_version += 1
    return _recognize(stored, cap.apply_corners(image, body.corners))


def _overlay_url(session: store.Session, image_id: str) -> str:
    version = session.images[image_id].overlay_version
    return f"/api/files/{session.session_id}/{image_id}/overlay.png?v={version}"


@router.get("/files/{session_id}/{image_id}/{name}")
def get_file(session_id: str, image_id: str, name: str):
    session = store.get_or_create(session_id)
    stored = session.images.get(image_id)
    if not stored:
        raise HTTPException(404, "not found")
    if name == "overlay.png" and stored.overlay_png:
        return Response(stored.overlay_png, media_type="image/png")
    if name == "original":
        return Response(stored.original, media_type=stored.content_type)
    raise HTTPException(404, "not found")
