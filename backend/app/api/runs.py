"""POST /runs: recognition → scene → pymunk simulation → formula verification."""

from fastapi import APIRouter, Depends, HTTPException

from app import store
from app.contracts import Run, RunRequest
from app.physics import formulas
from app.physics.simulate import simulate
from app.scene.builder import SceneError, build_scene

router = APIRouter()

COMPARED = ("angle_deg", "mass_kg", "gravity", "friction")


@router.post("/runs", response_model=Run)
def create_run(body: RunRequest, session: store.Session = Depends(store.session_dep)):
    stored = session.images.get(body.image_id)
    if not stored or not stored.recognition:
        raise HTTPException(404, "unknown image_id")
    try:
        built = build_scene(stored.recognition, body.params_override)
    except SceneError as exc:
        raise HTTPException(422, str(exc)) from exc
    scene = built.scene
    sim = simulate(scene)
    run_id, index = session.next_run_id()
    changed = {}
    if session.runs:
        old, new = session.runs[-1].scene.params, scene.params
        changed = {k: (getattr(old, k), getattr(new, k)) for k in COMPARED
                   if getattr(old, k) != getattr(new, k)}
    run = Run(
        run_id=run_id,
        index=index,
        scene=scene,
        frames=sim.frames,
        verification=formulas.verify(scene, sim.time_to_bottom, sim.final_speed),
        changed=changed,
    )
    session.runs.append(run)
    session.profile.runs = len(session.runs)
    session.log("run", run_id=run_id, image_id=body.image_id, warnings=built.warnings)
    if changed:
        session.log("change", run_id=run_id, changed=changed)
    return run
