"""Parameter validation and legacy physics handoff without collision-engine changes."""
import cv2
import numpy as np
import pytest

from app.contracts import DetectedObject, ParamsOverride
from app.cv.blocks import block_from_box
from app.cv.hybrid import VisionParameter
from app.cv.parameters import bind_parameters
from app.scene.builder import build_scene
from tests.test_hybrid import real_result


def reading(key, value, unit='', index=None, confidence=.99):
    return VisionParameter(key=key, value=value, unit=unit, text=f'{key}={value}{unit}',
                           object_index=index, confidence=confidence)


def bind(rec, readings):
    objects = [DetectedObject(kind='block', bbox=(0,0,100,100), confidence=.99,
                              body_id='block-1', geometry_verified=True),
               DetectedObject(kind='ball', bbox=(200,0,300,100), confidence=.99,
                              body_id='ball-1', geometry_verified=True)]
    bind_parameters(rec, readings, objects)


def test_primary_mass_and_global_labels_reach_existing_builder_without_copying_block_mass():
    rec = real_result().recognition
    bind(rec, [reading('mass_kg', 4, 'kg', 0), reading('mass_kg', 2, 'kg', 1),
               reading('angle_deg', 30, 'deg'), reading('friction', .2),
               reading('gravity', 9.8, 'm/s²'), reading('initial_speed_m_s', 3, 'm/s', 1)])
    block_mass, ball_mass, _, _, _, speed = rec.parameters
    assert block_mass.validated and not block_mass.applied
    assert block_mass.target == 'block-1' and ball_mass.target == 'ball-1'
    assert ball_mass.applied
    assert speed.validated and not speed.applied
    scene = build_scene(rec).scene
    assert scene.params.mass_kg == 2 and scene.params.angle_deg == 30
    assert scene.params.friction == .2 and scene.params.gravity == 9.8
    assert scene.params.sources['mass_kg'] == 'label'
    assert build_scene(rec, ParamsOverride(mass_kg=7)).scene.params.mass_kg == 7


def test_conflicting_ambiguous_unsupported_and_invalid_values_remain_visible():
    rec = real_result().recognition
    bind(rec, [reading('mass_kg', 4, 'kg', 1), reading('mass_kg', 5, 'kg', 1),
               reading('friction', .2, confidence=.7), reading('gravity', 9.8, 'kg'),
               reading('mass_kg', 4, 'kg', 99), reading('restitution', 2),
               reading('spring_constant', 10, 'N/m'), reading('mass_kg', 4, 'kg')])
    assert len(rec.parameters) == 8
    assert not any(p.validated or p.applied for p in rec.parameters)
    assert not rec.labels
    assert all(p.note for p in rec.parameters)


def test_units_are_normalized_and_only_the_matching_primary_receives_mass():
    rec = real_result().recognition
    bind(rec, [reading('mass_kg', 4000, 'g', 1)])
    assert rec.parameters[0].value == 4 and rec.parameters[0].unit == 'kg'
    assert rec.parameters[0].applied
    rec.selected_ball_index = 1
    rec.labels = []
    bind(rec, [reading('mass_kg', 4000, 'g', 1)])
    assert rec.parameters[0].validated and not rec.parameters[0].applied
    assert not rec.labels


def test_closed_quadrilateral_requires_real_ink_and_rejects_a_circle():
    mask = np.zeros((300,400), np.uint8)
    points = np.array([(100,80),(175,120),(135,195),(60,155)])
    cv2.polylines(mask, [points], True, 255, 4)
    block = block_from_box((55,75,180,200), mask, 3)
    assert block and len(block.vertices) == 4
    assert block.center == pytest.approx(points.mean(axis=0), abs=5)
    assert block_from_box((55,75,180,200), np.zeros_like(mask), 3) is None
    cv2.circle(mask, (300,140), 40, 255, 4)
    assert block_from_box((255,95,345,185), mask, 3) is None
