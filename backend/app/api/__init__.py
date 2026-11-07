from fastapi import APIRouter

from app.api import export, predict, runs, session, ta, upload

api_router = APIRouter(prefix="/api")
for module in (upload, runs, predict, session, ta, export):
    api_router.include_router(module.router)

public_router = export.public_router
