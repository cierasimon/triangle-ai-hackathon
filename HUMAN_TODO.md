# Human to-dos

Things only a human can do. Agents add lines here (what, why, which task it unblocks) and keep
building; nothing here blocks work.

- [ ] Confirm the measured angles in `fixtures/real/labels.json` (Checkpoint A filled
      `angle_deg` from the overlays: 25.4 / 37.0 / 57.7 / 23.6 / 28.0°, all
      `"angle_confirmed": false`). Check them against `make eval` overlays
      (`.venv/bin/python scripts/eval_cv.py --set real --overlays out/`), fix any, set
      `angle_confirmed` to true. Makes the real-photo angle error trustworthy.
- [ ] IMG_9141: say which leg is the ground. The recognizer reads the left line as ground
      (37°, consistent with the ruled lines); the written 52° only fits the bottom line.
      Both are accepted in `upright_rotation_k` until you decide.
- [ ] Take 3 ideal demo sketches (black marker, plain paper, upright, one ball on the ramp),
      plus photos of any failure cases (P1.2 tuning, demo).
- [ ] Try 2 live photos in `make dev` on your phone (P1.7).
- [ ] Drag the ramp on your phone (S2 / P2.2).
- [ ] Ask the TA a few questions; check tone per level (P3.3, P3.6).
- [ ] Scan the QR code 3 times on your phone (P4.3, P4.4): `brew install cloudflared`,
      `make serve` and `make tunnel` in two terminals, put the printed https URL in `.env` as
      `PUBLIC_BASE_URL`, restart `make serve`, open the tunnel URL, "Take Run N home", scan.
      Check the GIF plays and Download saves it. (Same-Wi-Fi alternative: open the app via
      the laptop's LAN IP; no tunnel needed.)
