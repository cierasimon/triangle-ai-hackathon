# Optional Gemini recognition

This backend-only update adds Gemini semantic proposals, ink-verified ball/block
geometry and handwritten parameter extraction to the existing OpenCV pipeline.
It preserves the existing single-ball simulation API and all frontend files.
It does not add collision dynamics or automatically simulate detected blocks.

## Setup

Use Python 3.12 and a project-specific virtual environment:

```sh
cd backend
python -m pip install -e '.[dev,vision]'
```

Set these variables in the ignored repository-root `.env` or process environment:

```dotenv
GOOGLE_CLOUD_PROJECT=your-google-cloud-project
GOOGLE_CLOUD_LOCATION=global
GEMINI_VISION_MODEL=gemini-3.5-flash-lite
```

The configured Vertex AI model must be available to the project and location.
Authenticate using Application Default Credentials or the deployment's service
identity. Never put credentials in the frontend or commit `.env`/credential files.
The Google SDK is optional: OpenCV-only installations do not import it.

## Frontend handoff

The existing upload request remains unchanged and defaults to local OpenCV.
To opt into Gemini, append one multipart form field:

```typescript
const form = new FormData();
form.append("photo", photo);
form.append("vision_mode", "hybrid");
const response = await fetch("/api/upload", {
  method: "POST", body: form, credentials: "include",
});
```

Hybrid mode transmits the processed image to Google Cloud and may incur API
charges. The frontend should disclose this before enabling it. Manual corner
correction retains the upload's recognition mode and can trigger another request.

The response retains `image_id`, `recognition`, `overlay_png_url` and every
existing recognition field. New fields in `recognition` are additive:

| Field | Meaning |
| --- | --- |
| `balls` | All confirmed circles in upright image pixels; legacy `ball` remains the primary |
| `selected_ball_index` | Index of the primary in `balls`, or null |
| `blocks` | Ink-supported quadrilaterals with `vertices`, `center` and confidence |
| `objects` | Semantic boxes with kind, confidence, `geometry_verified` and optional `body_id` |
| `parameters` | Readings with key, numeric value or null, SI unit, original text, target and status |
| `recognition_mode` | `opencv` or `hybrid` |
| `gemini_status` | `disabled`, `success`, `failed` or `unavailable` |
| `gemini_model`, `gemini_summary` | Model identifier and unverified reference summary |

`body_id`/parameter `target` use `ball-1`, `block-1`, etc. Indices are one-based
within their respective confirmed geometry lists; `scene` denotes a global
parameter and `unresolved` denotes an ambiguous or unverified association.
Boxes use `[left, top, right, bottom]`; all geometry shares the upright processed
image's pixel coordinate frame (`warped_size`). Parameters reference these IDs,
not the order of labels in the drawing or the order of semantic proposals.

Display parameter `text`, target, value/unit, confidence and status so users can
correct the model's reading. Confidence is model-reported, not measured accuracy.
For example, `m1=4kg` next to a block and `m2=4kg` next to a ball must produce two
separate readings targeting the corresponding block and ball, not a global mass.

### Parameter semantics and existing physics

- Extract every visible label, including unsupported quantities and symbolic
  unknowns. Use a null value for an unreadable number or unknown symbol.
- `validated=true` means confidence >= 0.85, unambiguous target, valid unit and
  supported numeric range. It does **not** mean the physics engine uses the value.
- `applied=true` means the reading was passed through the existing `labels` adapter:
  global angle/gravity/friction, or the primary ball's own mass. Existing
  `params_override` values still take priority when a run is created.
- Other bodies' masses, initial/angular speeds, restitution, duration and ramp
  length are returned for downstream integration with `applied=false`. This PR
  deliberately leaves the existing simulation, scaling and collision code unchanged.
- Unsupported quantities (forces, spring constants, dimensions beyond ramp length,
  etc.), conflicts, low confidence and invalid units remain visible with a reason.
- `applied` describes the default label handoff, not a guarantee of OCR correctness
  or the effective value after a user override. Inspect the run's parameter sources.

The overlay draws verified circles/quadrilaterals and labels unverified semantic
boxes as proposals. An unsupported proposal cannot create physics geometry on
blank paper. A missing ramp can be recovered only when both ramp and ground
proposals pass ink-support and intersection checks; supported existing lines are
preserved. Gemini failures leave the OpenCV result usable and add an explicit
warning. Recognition work runs off the asynchronous request event loop.

## Verification

```sh
cd backend
python -m pytest -q
```

The suite mocks the Gemini boundary: no credentials, paid requests, dataset
changes, retraining or held-out synthetic test data are needed. The existing CV
fixtures are regression checks, not a complete real-photo detection benchmark.
