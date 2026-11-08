"""POST /runs: recognition → scene → pymunk simulation → formula verification."""

from fastapi import APIRouter, Depends, HTTPException

from app import store
from app.contracts import ParamsOverride, Run, RunRequest
from app.physics import formulas
from app.physics.simulate import simulate, simulate_collisions
from app.scene.builder import SceneError, build_scene

router = APIRouter()

COMPARED = ("angle_deg", "mass_kg", "gravity", "friction")
CONCEPTS = {"mass_kg": "mass_independent", "angle_deg": "angle_effect",
            "gravity": "gravity_effect", "friction": "friction_effect"}


def make_run(session: store.Session, image_id: str, override: ParamsOverride | None,
             reason: str = "change") -> Run:
    """Build, simulate, verify and record a run. Raises SceneError / KeyError."""
    stored = session.images[image_id]
    built = build_scene(stored.recognition, override)
    scene = built.scene
    sim = simulate(scene)
    motion = simulate_collisions(scene)  # the animation: down the ramp and along the ground
    run_id, index = session.next_run_id()
    changed = {}
    motion_changed = False
    if session.runs:
        old, new = session.runs[-1].scene.params, scene.params
        changed = {k: (getattr(old, k), getattr(new, k)) for k in COMPARED
                   if getattr(old, k) != getattr(new, k)}
        motion_changed = old.motion != new.motion
    run = Run(
        run_id=run_id,
        index=index,
        scene=scene,
        frames=motion.frames,
        verification=formulas.verify(scene, sim.time_to_bottom, sim.final_speed),
        changed=changed,
        collision=motion.collision,
    )
    session.runs.append(run)
    profile = session.profile
    profile.runs = len(session.runs)
    for key in changed:
        profile.concepts_seen[CONCEPTS[key]] = True
    if motion_changed:
        profile.concepts_seen["roll_effect"] = True
    session.log("run", run_id=run_id, image_id=image_id, warnings=built.warnings)
    if changed or motion_changed:
        kind = {"drag": "drag_rerun", "quiz": "quiz"}.get(reason, "change")
        data = {"run_id": run_id, "changed": changed}
        if motion_changed:
            data["motion"] = scene.params.motion
        session.log(kind, **data)
    return run


@router.post("/runs", response_model=Run)
def create_run(body: RunRequest, session: store.Session = Depends(store.session_dep)):
    stored = session.images.get(body.image_id)
    if not stored or not stored.recognition:
        raise HTTPException(404, "unknown image_id")
    try:
        return make_run(session, body.image_id, body.params_override, body.reason)
    except SceneError as exc:
        raise HTTPException(422, str(exc)) from exc
