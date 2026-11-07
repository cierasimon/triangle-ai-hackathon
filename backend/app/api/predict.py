"""GET /predict (formula only, fast) and POST /whycard (placeholder until P2.4)."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException

from app import store
from app.contracts import PredictResponse, WhyCard, WhyCardRequest
from app.physics import formulas

router = APIRouter()


@router.get("/predict", response_model=PredictResponse)
def predict(
    angle_deg: float,
    length_m: float,
    mass_kg: float = 1.0,
    gravity: float = 9.81,
    friction: float = 0.0,
    motion: Literal["slide", "roll"] = "slide",
):
    p = formulas.predict(angle_deg, length_m, gravity, friction, motion, mass_kg)
    return PredictResponse(time_s=p.time_s, speed_m_s=p.speed_m_s, will_move=p.will_move,
                           note=p.note)


@router.post("/whycard", response_model=WhyCard)
def whycard(body: WhyCardRequest, session: store.Session = Depends(store.session_dep)):
    a, b = session.get_run(body.run_a), session.get_run(body.run_b)
    if not a or not b:
        raise HTTPException(404, "unknown run")
    return WhyCard(
        latex=r"t = \sqrt{\dfrac{2L}{g\sin\theta}}",
        symbol="m",
        sentence="FAKE DATA: mass does not appear in the formula, so it cancels.",
        cancels=True,
    )
