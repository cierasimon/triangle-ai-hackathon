PY      := .venv/bin/python
UVICORN := .venv/bin/uvicorn
PYTEST  := .venv/bin/pytest
SET     ?= all

.PHONY: setup dev backend frontend test eval data fixtures build tunnel

setup:  ## venv (Python 3.12 via uv) + backend deps + frontend deps
	uv venv --python 3.12 --allow-existing
	uv pip install -e "backend[dev]"
	cd frontend && npm ci

dev:    ## backend on :8000 (reload) + Vite on :5173; open http://localhost:5173
	@trap 'kill 0' INT TERM EXIT; \
	$(UVICORN) app.main:app --app-dir backend --reload --reload-dir backend --host 0.0.0.0 --port 8000 & \
	npm --prefix frontend run dev & \
	wait

backend:
	$(UVICORN) app.main:app --app-dir backend --reload --reload-dir backend --host 0.0.0.0 --port 8000

frontend:
	npm --prefix frontend run dev

test:   ## pytest (no live LLM) + frontend typecheck
	cd backend && ../$(PYTEST)
	npm --prefix frontend run typecheck

eval:   ## recognition accuracy; make eval SET=synthetic|real|all
	$(PY) scripts/eval_cv.py --set $(SET)

data:   ## download the synthetic dataset into data/generated/
	./scripts/data/fetch_dataset.sh

fixtures:  ## rebuild fixtures/synthetic from the downloaded dataset
	$(PY) scripts/data/make_fixtures.py --count 20 \
	  --source $(dir $(firstword $(wildcard data/generated/*/annotations.jsonl data/generated/annotations.jsonl)))

build:  ## build the frontend into frontend/dist (served by the backend at /)
	npm --prefix frontend run build

tunnel: ## public URL for :8000 (brew install cloudflared); then set PUBLIC_BASE_URL
	cloudflared tunnel --url http://localhost:8000
