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

See [AGENTS.md](AGENTS.md) for all commands, rules and the repo layout. The synthetic dataset
is described in [data/synthetic_v3/README.md](data/synthetic_v3/README.md).

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

