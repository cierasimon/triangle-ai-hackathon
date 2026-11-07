cd ..# PaperPhysics V3 — Synthetic Inclined-Plane Dataset (3,000 images)

This dataset contains 3,000 generated, hand-drawn-style physics diagrams for
**PaperPhysics**. It is intended for computer-vision object detection, diagram
parsing, parameter OCR, and object-parameter association—not as a standalone
dataset for learning physics trajectories.

## What's new in V3

1. **Blocks align with their supporting surface.** On a slope, both bottom
   corners lie on the inclined line and the block's bottom edge is parallel to
   that line. On horizontal ground, its bottom edge is horizontal.
2. **Parameter text is contextual** instead of appearing in one fixed table:
   angle near the ramp corner; mass near the object; speed near the direction
   arrow; kinetic friction near the incline; spring constant near the spring;
   target mass and restitution near the collision target.
3. **Collision targets** may rest on either the incline or horizontal ground.
   Both target types (ball and block) are included.
4. **Annotation relationships** preserve which text refers to which object or
   arrow, including geometry, contact edge, and whether the value is actually
   printed on the image.
5. Unicode-capable font selection with ASCII fallback for systems that cannot
   render `θ`, `μ`, or `°` using their default Pillow fonts.

## Exact distribution (default 3,000-image run)

| Scenario | Count |
| --- | ---: |
| Ball + incline | 375 |
| Block + incline | 375 |
| Ball + incline + spring | 375 |
| Block + incline + spring | 375 |
| Ball + incline + collision target on incline | 375 |
| Ball + incline + collision target on flat | 375 |
| Block + incline + collision target on incline | 375 |
| Block + incline + collision target on flat | 375 |

The collision target's **shape** is independently randomized between ball and
block. `collision_target_surface` is `ramp` or `flat` for collision scenes and
`null` otherwise.

## Files

```text
paperphysics_v3_3000/
├── README.md
├── images/                 # 3000 RGB PNG images (768 × 576 by default)
├── labels/                 # YOLO-format axis-aligned bounding boxes
├── annotations.jsonl       # One rich record per image
├── classes.txt             # ball, block, ramp, spring
├── dataset.yaml            # YOLO metadata and split file paths
├── train.txt               # 80% stratified within scenario
├── val.txt                 # Approximately 10%
├── test.txt                # Approximately 10%
└── manifest.json           # Counts, dimensions, seed
```

The accompanying zip also contains `paperphysics_generate_v3.py` and
`paperphysics_v3_preview.jpg`.

## Reproduce on macOS / Windows / Linux

Python 3.10+ and Pillow are sufficient:

```bash
python -m pip install pillow
python paperphysics_generate_v3.py --count 3000 --output paperphysics_v3_3000
```

You can also use `--out` in place of `--output`, and customize `--seed`,
`--width`, and `--height`. Running again with the same seed and parameters
produces identical annotation geometry and scene choices. Output PNG files
may differ across Pillow versions because font and rasterization differ.
Generate into a fresh output directory to avoid leftover files from a previous
run with a different `--count`.

## Object categories and geometry

YOLO class indices:

```text
0 ball
1 block
2 ramp
3 spring
```

YOLO labels contain normalized `class cx cy width height`. Rich `annotations.jsonl`
records also contain object IDs, roles (`primary`, `collision_target`, etc.),
object polygons, boxes in pixels, and supporting surfaces.

For block objects, `contact_edge_xy` contains the exact two corners of the
bottom edge. In `ramp` cases, it is parallel to the incline. For `flat`
cases, it is horizontal. `block_angle_radians` is the screen-coordinate angle
of the supporting edge, not a separate physical orientation measurement.

## Parameters and relationships

The following fields are among those in `physics`:

| Key | Meaning |
| --- | --- |
| `angle_degrees` | Ramp inclination θ |
| `mass1_kg` | Primary object's mass |
| `initial_speed_m_s` | Primary initial speed (scalar) |
| `motion_direction` | `down_slope`, `up_slope`, or `stationary` |
| `kinetic_friction_coefficient` | μ_k |
| `gravity_m_s2` | Assumed gravitational acceleration |
| `mass2_kg` | Collision target mass, if present |
| `target_initial_speed_m_s` | Target speed, normally 0 |
| `restitution` | Coefficient of restitution e |
| `spring_constant_N_m` | Spring stiffness k |
| `spring_compression_m` | Spring compression x0 |

`physics` numeric parameters have the structure:

```json
"mass1_kg": {"value": 2.1, "shown_on_image": true}
```

The generator *knows* every randomly selected scene parameter, but some are
**not printed**. This distinction is crucial: only parameters with a matching
`text_annotations` entry are valid OCR supervision targets.

`text_annotations` contains:

- `parameter_key`, parsed numeric `value`, exact printed `text`;
- text `bbox_xyxy` in image pixels;
- `related_object_id` (references an object or arrow ID, or `null` for global
  gravity);
- `placement`, such as `angle_at_corner`, `next_to_arrow`,
  `next_to_primary`, `next_to_spring`, or `collision_parameter_near_target`.

This allows evaluation of **text recognition and parameter-to-object
association** separately.

## Important limitations

- Shapes imitate hand sketches, but **text is typeset**, not written by a human
  on an iPad. Validate on independent real sketches with multiple writers.
- The diagrams show **initial configurations**, not physically simulated
  trajectories. A collision target's presence does not prove a collision will
  occur given the initial velocity and friction.
- Spring compression and stiffness values are illustrative physical inputs;
  the drawn spring's pixel length is not a calibrated meter-scale measurement.
- Ramp length is in image pixels. There is no inferred meters-per-pixel scale.
- Kinetic friction is not the same as static friction; no static friction is
  assigned unless you add it separately to a simulation.
- Labels may be absent from a given image. Never treat an unprinted value as
  visually inferable ground truth.
- Although train/val/test are separated by images and stratified by scenario,
  they all use **the same synthetic generator**. This is useful for a pipeline
  smoke test but can greatly overestimate generalization to real drawings.

## Training suggestion

1. Start YOLO on `ball`, `block`, `ramp`, `spring` detection.
2. Use a text detector/OCR model to identify the `text_annotations` spans.
3. Train or evaluate grounding from parameter text to `related_object_id`.
4. Feed geometry, physical parameters, and visible/unknown flags to Pymunk or
   another deterministic physics engine.
5. Hold out 50–100 real iPad sketches from **different people** for a genuine
   benchmark of the entire image → JSON → simulation pipeline.
