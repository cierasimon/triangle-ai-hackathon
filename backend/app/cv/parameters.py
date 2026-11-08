"""Normalize labelled SI values and apply only confident, unambiguous supported inputs."""
import math

from app.contracts import DiagramParameter, Label

# (minimum, maximum, canonical SI unit). Other readable quantities remain visible.
LIMITS = {
    'mass_kg': (.01, 10000, 'kg'), 'initial_speed_m_s': (-10, 10, 'm/s'),
    'angular_speed_rad_s': (-100, 100, 'rad/s'), 'angle_deg': (0, 90, 'deg'),
    'gravity': (0, 100, 'm/s²'), 'friction': (0, 5, ''),
    'restitution': (0, 1, ''), 'duration_s': (.1, 5, 's'),
    'ramp_length_m': (.01, 100, 'm'),
}
UNITS = {'°':'deg', 'degrees':'deg', 'degree':'deg', 'm/s^2':'m/s²', 'm/s2':'m/s²',
         'dimensionless':'', '1':'', 'g':'g', 'cm':'cm'}
CONVERSIONS = {('mass_kg','g'): .001, ('ramp_length_m','cm'): .01,
               ('initial_speed_m_s','cm/s'): .01}


def bind_parameters(rec, parameters, objects):
    rec.parameters = []
    for p in parameters:
        obj = objects[p.object_index] if p.object_index is not None and 0 <= p.object_index < len(objects) else None
        target = ('scene' if p.object_index is None or obj and obj.kind in {'ramp','platform'}
                  else obj.body_id if obj else None)
        item = DiagramParameter(key=p.key, value=p.value, unit=p.unit.strip(), text=p.text,
                                target=target or 'unresolved', confidence=p.confidence)
        item.unit = UNITS.get(item.unit,item.unit)
        if (item.key,item.unit) in CONVERSIONS and item.value is not None:
            item.value *= CONVERSIONS[(item.key,item.unit)]
            item.unit = LIMITS[item.key][2]
        if item.key not in LIMITS:
            item.note = 'Read from the diagram; not supported by the current physics engine and not applied.'
        elif item.value is None or item.confidence < .85:
            item.note = 'The value is unclear or confidence is too low; confirmation is required.'
        elif not target:
            item.note = 'No reliable association with a verified body; not applied.'
        else:
            low,high,unit = LIMITS[item.key]
            global_key = item.key in {'angle_deg','gravity','restitution','duration_s','ramp_length_m'}
            body_key = item.key in {'mass_kg','initial_speed_m_s','angular_speed_rad_s'}
            if item.unit != unit or not low <= item.value <= high:
                item.note = 'Unit or value is outside the supported range; not applied.'
            elif global_key and target != 'scene' or body_key and target == 'scene':
                item.note = 'The parameter target is ambiguous; not applied.'
            else:
                item.validated = True
                item.note = 'The reading passed validation; the current single-ball engine does not apply this parameter.'
        rec.parameters.append(item)
    groups = {}
    for p in rec.parameters:
        if p.validated:
            groups.setdefault((p.target,p.key),[]).append(p)
    for group in groups.values():
        if max(p.value for p in group)-min(p.value for p in group) > 1e-6:
            for p in group:
                p.validated=False; p.note='Conflicting labels for the same parameter; confirmation is required and neither value is applied.'
    # Minimal adapter for main's existing single-ball builder. Do not alter dynamics.
    # A primary-ball mass applies only to that ball; never to another recognized body.
    primary = f"ball-{rec.selected_ball_index+1}" if rec.selected_ball_index is not None else None
    for p in rec.parameters:
        supported_global = p.target == 'scene' and p.key in {'angle_deg', 'gravity', 'friction'}
        primary_body = p.target == primary and p.key in {'mass_kg', 'initial_speed_m_s'}
        if p.validated and (supported_global or primary_body):
            rec.labels = [label for label in rec.labels if label.key != p.key]
            rec.labels.append(Label(key=p.key, value=p.value, text=p.text, confidence=p.confidence))
            p.applied = True
            p.note = 'Passed to the existing single-ball engine; user overrides take priority.'
    # Initial speed of the other bodies (B, C, ...), and mass / initial speed of a main body that is a
    # block (the code above only knows a main ball): match a parameter's body to a simulated body by position.
    centers = {f"ball-{i + 1}": b.center for i, b in enumerate(rec.balls)}
    centers |= {f"block-{i + 1}": b.center for i, b in enumerate(rec.blocks)}
    rec.other_speeds = [0.0] * len(rec.other_balls)

    def tol(ball):
        return max(0.8 * ball.radius, 4)

    def give_main(p):
        rec.labels = [label for label in rec.labels if label.key != p.key]
        rec.labels.append(Label(key=p.key, value=p.value, text=p.text, confidence=p.confidence))
        p.applied, p.note = True, 'Passed to the simulation for the main body.'

    for p in rec.parameters:
        center = centers.get(p.target)
        if not (p.validated and center is not None and p.value is not None):
            continue
        if p.key == 'initial_speed_m_s':
            for j, other in enumerate(rec.other_balls):
                if math.dist(center, other.center) <= tol(other):
                    rec.other_speeds[j] = p.value
                    p.applied, p.note = True, 'Passed to the simulation for this body.'
        if (p.key in {'mass_kg', 'initial_speed_m_s'} and not p.applied and rec.ball is not None
                and rec.selected_ball_index is None and math.dist(center, rec.ball.center) <= tol(rec.ball)):
            give_main(p)

    # A picture with exactly one body: its mass or speed label cannot belong to anything else, even
    # if the model did not tie the label to the body (it sits away from it, e.g. "M = 6 kg").
    if rec.ball is not None and not rec.other_balls:
        for key in ('mass_kg', 'initial_speed_m_s'):
            readings = [p for p in rec.parameters if p.key == key and p.value is not None]
            low, high, unit = LIMITS[key]
            if (len(readings) == 1 and not readings[0].applied and readings[0].confidence >= .85
                    and readings[0].unit == unit and low <= readings[0].value <= high
                    and readings[0].target in {'scene', 'unresolved'}):
                give_main(readings[0])
                readings[0].validated = True
