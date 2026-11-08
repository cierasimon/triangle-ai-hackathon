# PaperPhysics

Draw a ramp and a ball on paper → take a photo → it becomes a running simulation, checked
against the textbook formula → change the ramp, mass or gravity and see why → ask an AI TA →
scan a QR code to take the animation home.

> **Status: Phases 1–4 done.** Photo → recognition → pymunk simulation → formula check,
> then change angle/mass/gravity/friction/motion, drag the ramp, compare runs, read the
> why-card, ask the TA (free OpenRouter model, or offline templates with no key), and take
> the animation home as a GIF via a QR code. See [plan.md](plan.md).

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

See [Makefile](Makefile) for commands and [backend/RECOGNITION.md](backend/RECOGNITION.md)
for the recognition API. The synthetic dataset is described in
[data/synthetic_v3/README.md](data/synthetic_v3/README.md).

## Gemini image recognition

The upload page includes a **Use Gemini + OpenCV recognition** checkbox, enabled by
default. The notice explains that uploading in this mode sends the processed image
to Google Cloud and may incur API charges. Turn it off for fully local OpenCV.
The API itself still defaults to OpenCV for existing clients.

Install the optional SDK into the existing project environment:

```sh
make setup-vision
# For the existing backend/.venv layout:
make setup-vision VENV=backend/.venv
```

Set `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` and `GEMINI_VISION_MODEL` using
`.env.example`, and configure Google Application Default Credentials or a service
identity with access to the selected Vertex AI model. Credentials stay on the backend.
Missing configuration or a failed request is shown explicitly as an OpenCV fallback.

The recognition panel shows all confirmed balls/blocks and each handwritten
parameter's original text, body association and application status. Ambiguous or
repeated object associations require confirmation. The current simulation still
uses **one primary ball**; block and multi-object collision dynamics are not part
of this update. Read-only parameters are not silently treated as simulation inputs.

Local diagnostic reports and overlays are saved under ignored `data/generated/`:

```sh
make inspect-recognition
# With an existing backend environment:
make inspect-recognition VENV=backend/.venv
# Optional paid comparison, only when explicitly requested:
.venv/bin/python scripts/inspect_recognition.py --mode both --allow-cloud
```

These reports do not calculate real-photo accuracy. Existing OpenCV benchmark
checks remain in `scripts/eval_cv.py`; complete verified annotations are needed
before claiming multi-object detection or parameter-reading accuracy.
GitHub Actions runs offline backend tests, frontend tests and the frontend build
on pull requests and pushes to `main`, without Google or OpenRouter secrets.

## Take it home on a phone

"Take Run N home" renders a GIF on the server and shows a QR code to a small download page
(`/s/<id>`). Exports live in memory until the server restarts.

- **Same Wi-Fi:** run `make dev`, open `http://<your-laptop-ip>:5173` on the laptop, and the
  QR code points at that address, so a phone on the same network can open it.
- **Anywhere (tunnel):** `brew install cloudflared`, then in two terminals:

  ```bash
  make serve
  ```

  ```bash
  make tunnel
  ```

  Copy the `https://….trycloudflare.com` URL it prints into `.env` as `PUBLIC_BASE_URL`,
  restart `make serve`, and open the tunnel URL. QR codes now point at the tunnel.
  (`PUBLIC_BASE_URL` wins whenever it isn't the localhost default; otherwise the address the
  app was opened on is used.)

