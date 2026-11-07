# PaperPhysics

Draw a ramp and a ball on paper → take a photo → it becomes a running simulation, checked
against the textbook formula → change the ramp, mass or gravity and see why → ask an AI TA →
scan a QR code to take the animation home.

> **Status: Phase 1 done.** Photo → recognition → pymunk simulation → formula check is
> real. Changes, drag, why-cards, the TA and GIF export are next ([plan.md](plan.md));
> those routes still return placeholder data marked "FAKE".

## Run it

Needs [uv](https://docs.astral.sh/uv/) and Node ≥ 20.

```bash
make setup
make dev
```

Open http://localhost:5173 (on a phone on the same Wi-Fi: `http://<your-laptop-ip>:5173`),
choose a photo (try one from `fixtures/synthetic/`) and watch the run play.

```bash
make test
```

Optional: `cp .env.example .env` and add an OpenRouter key (free models only). Without a key
the app runs in template mode.

See [AGENTS.md](AGENTS.md) for all commands, rules and the repo layout. The synthetic dataset
is described in [data/synthetic_v3/README.md](data/synthetic_v3/README.md).
