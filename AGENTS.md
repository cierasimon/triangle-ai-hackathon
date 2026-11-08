# AGENTS.md

PaperPhysics: draw a ramp and a ball on paper → photo → verified simulation → change it →
ask an AI TA → take the animation home. Full task list: [plan.md](plan.md).

## Ground rules

1. **Scope is one situation: a ball on a ramp** (plus a flat ground line). No blocks, springs,
   collisions, pendulums or velocity arrows. A situation is a plug-in (detector + scene builder
   + formulas + why-cards), but only `ball_on_ramp` is built.
2. **Never break the demo.** After every merge, `make dev` must still run photo → simulation.
   Half-built features go behind a flag.
3. **Stay in your folder.** Each task in plan.md lists the folders it owns. Touch files outside
   them only to wire things up, and say so in the commit message.
4. **Contracts are law.** Modules talk only through the models in `backend/app/contracts.py`,
   mirrored in `frontend/src/types.ts`. Changing a contract is its own small commit that
   updates every user of it (both files).
5. **Prove "done" with a command.** Every task has acceptance checks; add them as `pytest`
   tests or a script, and run them before calling the task done.
6. **No paid API calls, ever.** LLM model ids must end in `:free` or be `openrouter/free`; the
   LLM client refuses anything else.
7. **Never read, print or commit `.env`.** Document variables in `.env.example`.
8. **The physics engine is the only source of numbers.** The LLM may phrase things, never
   compute them.
9. **Don't stop for the human; leave a note.** Things only a human can do (confirming angles,
   more photos, phone tests) go in [HUMAN_TODO.md](HUMAN_TODO.md) (what, why, which task it
   unblocks). Never fake the human's part; label unconfirmed numbers "unconfirmed".

## Run and test

Requires `uv` and Node ≥ 20. Python is 3.12 in `.venv` (the system Python 3.9 is not used).

```bash
make setup   # uv venv --python 3.12, uv pip install -e backend[dev], npm ci in frontend
make dev     # backend :8000 (reload) + Vite :5173 → open http://localhost:5173
make test    # pytest (live LLM test skipped) + frontend typecheck + vitest
make eval    # scripts/eval_cv.py (SET=synthetic|real|all)
make data    # download the 3,000-image synthetic dataset into data/generated/ (gitignored)
make build   # frontend → frontend/dist, served by the backend at http://localhost:8000/
make serve   # build UI + serve UI and API on :8000 (one process, for the tunnel / demo)
make tunnel  # cloudflared quick tunnel to :8000; put the URL in PUBLIC_BASE_URL
```

- Single backend test: `cd backend && ../.venv/bin/pytest tests/test_api.py -k predict`
- Opt-in real LLM test (20 scripted questions, ~3 min, reads `.env`):
  `cd backend && ../.venv/bin/pytest -m live_llm -s`
- `make dev` uses the key in `.env`; `LLM_PROVIDER=none make dev` forces template mode.
- Tests set `LLM_PROVIDER=none` and never read `.env`; mock `ta/llm.py` in tests.

## Layout

- `backend/app/contracts.py` – every shared pydantic model and API body.
- `backend/app/store.py` – in-memory sessions (cookie `pp_session` or header `X-Session`),
  runs, events, uploaded files, shares.
- `backend/app/api/` – routers. Photo → simulation is real (Phase 1): `upload.py` runs
  `cv/capture.py` → `cv/recognize.py`; `runs.py` runs `scene/builder.py` →
  `physics/simulate.py` → `physics/formulas.py`; `/whycard` uses `physics/whycard.py`.
  `ta.py` → `ta/assistant.py` (LLM or `ta/templates.py`, then `ta/numcheck.py`),
  `ta/summary.py`, `ta/followups.py`, `ta/quiz.py`; context in `ta/context.py`, client in
  `ta/llm.py` (free models only). `export.py` → `export/gif.py` (Pillow GIF + card, PNG
  fallback), share page `/s/{id}` and QR; share base URL logic in `export.public_base`.
- `backend/app/cv/` – capture (paper warp, ink mask) and recognition (lines, ball,
  orientation). Tune with `make eval` and `scripts/eval_cv.py --overlays out/ -v`.
- `backend/app/physics/simulate.py` – the S1 roll-vs-slide result is at the top.
- `frontend/src/` – `types.ts` (contracts mirror), `api.ts` (typed client), `components/`.
  `physics.ts` ports `formulas.predict` for live drag updates; both sides are pinned by
  `predict_golden.json` (after a formula change: `cd backend && REGEN_GOLDEN=1
  ../.venv/bin/pytest tests/test_predict_golden.py`, then `npm --prefix frontend test`).
- `fixtures/synthetic/` – 20 `ball_plain` images + their annotation JSON (rebuild with
  `make data && make fixtures`). `fixtures/real/` – phone photos + `labels.json`.
- `scripts/data/` – dataset generator, fetch script, fixture builder.
