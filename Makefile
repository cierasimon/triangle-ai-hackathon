VENV    ?= .venv
VENV_ABS := $(abspath $(VENV))
PY      := $(VENV_ABS)/bin/python
UVICORN := $(VENV_ABS)/bin/uvicorn
PYTEST  := $(VENV_ABS)/bin/pytest
SET     ?= all

.PHONY: setup setup-vision inspect-recognition dev backend frontend serve test eval data fixtures build tunnel

setup:  ## venv (Python 3.12 via uv) + backend deps + frontend deps
	uv venv "$(VENV)" --python 3.12 --allow-existing
	uv pip install --python "$(PY)" -e "backend[dev]"
	cd frontend && npm ci

setup-vision:  ## install optional Gemini SDK into an existing project environment
	uv pip install --python "$(PY)" -e "backend[dev,vision]"

inspect-recognition:  ## local fixture diagnostics; no cloud requests by default
	$(PY) scripts/inspect_recognition.py

dev:    ## backend on :8000 (reload) + Vite on :5173; open http://localhost:5173
	@trap 'kill 0' INT TERM EXIT; \
	$(UVICORN) app.main:app --app-dir backend --reload --reload-dir backend --host 0.0.0.0 --port 8000 & \
	npm --prefix frontend run dev & \
	wait

backend:
	$(UVICORN) app.main:app --app-dir backend --reload --reload-dir backend --host 0.0.0.0 --port 8000

frontend:
	npm --prefix frontend run dev

serve:  ## demo mode: build the UI, serve UI + API from one process on :8000 (for the tunnel)
	npm --prefix frontend run build
	$(UVICORN) app.main:app --app-dir backend --host 0.0.0.0 --port 8000

test:   ## pytest (no live LLM) + frontend typecheck + vitest
	cd backend && "$(PYTEST)"
	npm --prefix frontend run typecheck
	npm --prefix frontend test

eval:   ## recognition accuracy; make eval SET=synthetic|real|all
	$(PY) scripts/eval_cv.py --set $(SET)

data:   ## download the synthetic dataset into data/generated/
	./scripts/data/fetch_dataset.sh

fixtures:  ## rebuild fixtures/synthetic from the downloaded dataset
	$(PY) scripts/data/make_fixtures.py --count 20 \
	  --source $(dir $(firstword $(wildcard data/generated/*/annotations.jsonl data/generated/annotations.jsonl)))

build:  ## build the frontend into frontend/dist (served by the backend at /)
	npm --prefix frontend run build

tunnel: ## public https URL for :8000 (brew install cloudflared); run `make serve` first
	@command -v cloudflared >/dev/null || { echo "cloudflared not found: brew install cloudflared"; exit 1; }
	cloudflared tunnel --url http://localhost:8000
