"""Offline hybrid tests mock the remote boundary; no cloud requests in unit tests."""
import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from app.contracts import Recognition
from app.cv import capture as cap
from app.cv import hybrid
from app.cv.recognize import Result, recognize
from tests.conftest import FIXTURES


def real_result():
    path = FIXTURES / "real/IMG_9141.jpg"
    return recognize(cap.capture(cap.load_image(path.read_bytes())), path.stem)


def semantic_balls(result):
    w, h = result.recognition.warped_size
    objects = []
    for b in result.recognition.balls:
        x, y = b.center
        objects.append({"kind": "ball", "confidence": .9,
                        "bbox": [(x-b.radius)*1000/w, (y-b.radius)*1000/h,
                                 (x+b.radius)*1000/w, (y+b.radius)*1000/h]})
    return hybrid.VisionAnalysis(objects=objects, ramp=None, ground=None, summary="Two balls and a ramp.")


def test_existing_real_two_ball_photo_returns_both_circles():
    result = real_result()
    assert len(result.recognition.balls) == 2
    assert result.recognition.ball == result.recognition.balls[0]


def test_hybrid_preserves_geometry_and_both_balls(monkeypatch):
    result = real_result()
    before = [l.model_dump() for l in result.recognition.lines]
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "unit-test")
    monkeypatch.setattr(hybrid, "analyze", lambda *_: semantic_balls(result))
    merged = hybrid.assist(result)
    assert merged.recognition.gemini_status == "success"
    assert len(merged.recognition.balls) == 2
    assert all(o.geometry_verified for o in merged.recognition.objects)
    assert [l.model_dump() for l in merged.recognition.lines] == before


def test_remote_failure_retains_working_cv_and_hides_credentials(monkeypatch):
    result = real_result()
    before = [b.model_dump() for b in result.recognition.balls]
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "unit-test")
    def fail(*_):
        raise TimeoutError("SECRET credential contents must not reach the browser")
    monkeypatch.setattr(hybrid, "analyze", fail)
    merged = hybrid.assist(result)
    assert merged.recognition.gemini_status == "failed"
    assert [b.model_dump() for b in merged.recognition.balls] == before
    assert "SECRET" not in merged.recognition.model_dump_json()


def test_unconfigured_hybrid_is_explicitly_local_only(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    result = hybrid.assist(real_result())
    assert result.recognition.gemini_status == "unavailable"


def blank_result(with_lines=False):
    image = np.full((300, 400, 3), 255, np.uint8)
    if with_lines:
        cv2.line(image, (70, 60), (300, 230), (0, 0, 0), 4)
        cv2.line(image, (40, 230), (350, 230), (0, 0, 0), 4)
    mask = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 128).astype(np.uint8) * 255
    return Result(Recognition(image_id="test", paper_found=False, paper_corners=None,
                              warped_size=(400, 300), lines=[], ball=None), image, mask, 0, None,
                  {"stroke": 4.})


def test_semantic_proposal_cannot_invent_geometry_on_blank_paper():
    analysis = hybrid.VisionAnalysis(objects=[{"kind": "ball", "bbox": [100, 100, 220, 250], "confidence": .99}],
                                    ramp={"p1": [175, 200], "p2": [750, 767]},
                                    ground={"p1": [100, 767], "p2": [875, 767]}, summary="untrusted")
    result = hybrid.fuse(blank_result(), analysis)
    assert not result.recognition.lines and not result.recognition.balls
    assert not result.recognition.objects[0].geometry_verified


def test_ink_supported_gemini_line_can_recover_missing_ramp():
    analysis = hybrid.VisionAnalysis(objects=[], ramp={"p1": [175, 200], "p2": [750, 767]},
                                    ground={"p1": [100, 767], "p2": [875, 767]}, summary="ramp")
    result = hybrid.fuse(blank_result(with_lines=True), analysis)
    assert {l.role for l in result.recognition.lines} == {"ramp", "ground"}
    ramp = result.recognition.lines[0]
    assert np.linalg.norm(np.array(ramp.p1) - [70, 60]) < 6
    assert np.linalg.norm(np.array(ramp.p2) - [300, 230]) < 6


def test_malformed_geometry_is_rejected():
    with pytest.raises(ValidationError):
        hybrid.VisionAnalysis(objects=[{"kind": "ball", "bbox": [500, 500, 100, 100], "confidence": .8}],
                              ramp=None, ground=None, summary="bad")


def test_google_wire_schema_stays_simple_while_local_validation_is_strict():
    import json
    encoded = json.dumps(hybrid.response_schema())
    assert "prefixItems" not in encoded and "$ref" not in encoded
    with pytest.raises(ValidationError):
        hybrid.VisionAnalysis(objects=[{"kind": "ball", "bbox": [1, 2, 3], "confidence": .8}],
                              ramp=None, ground=None, summary="wrong coordinate length")


def test_hybrid_upload_reaches_remote_boundary_offline(client, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "unit-test")
    calls = []
    def fake(image, result):
        calls.append(image.shape)
        return semantic_balls(result)
    monkeypatch.setattr(hybrid, "analyze", fake)
    path = FIXTURES / "real/IMG_9141.jpg"
    response = client.post("/api/upload", data={"vision_mode": "hybrid"},
                           files={"photo": (path.name, path.read_bytes(), "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["recognition"]["gemini_status"] == "success"
    assert len(calls) == 1


def two_nearby_balls():
    from app.contracts import Ball
    result = blank_result()
    result.recognition.balls = [Ball(center=(120,100),radius=20,confidence=.99),
                                Ball(center=(160,100),radius=20,confidence=.99)]
    result.recognition.ball = result.recognition.balls[0]
    for ball in result.recognition.balls:
        cv2.circle(result.mask,tuple(map(int,ball.center)),20,255,3)
    return result


@pytest.mark.parametrize('same_mass', [True, False])
def test_box_covering_multiple_balls_cannot_bind_mass(same_mass):
    result = two_nearby_balls()
    analysis = hybrid.VisionAnalysis(objects=[{'kind':'ball','bbox':[225,233,475,433],'confidence':.99}]*2,
        ramp=None, ground=None, summary='ambiguous', parameters=[
            {'key':'mass_kg','value':4,'unit':'kg','text':'m1=4kg','object_index':0,'confidence':.99},
            {'key':'mass_kg','value':4 if same_mass else 2,'unit':'kg','text':'m2','object_index':1,'confidence':.99}])
    rec = hybrid.fuse(result,analysis).recognition
    assert len(rec.balls) == 2  # Preserve valid CV geometry even when semantics are ambiguous.
    assert all(obj.body_id is None for obj in rec.objects)
    assert all(p.target == 'unresolved' and not p.validated and not p.applied for p in rec.parameters)
    assert not rec.labels


def test_duplicate_proposals_for_one_ball_invalidate_both_associations():
    result = real_result()
    analysis = semantic_balls(result)
    analysis.objects = [analysis.objects[0], analysis.objects[0].model_copy()]
    analysis.parameters = [hybrid.VisionParameter(key='mass_kg',value=4,unit='kg',text='4kg',
                                                object_index=i,confidence=.99) for i in range(2)]
    rec = hybrid.fuse(result,analysis).recognition
    assert all(obj.body_id is None for obj in rec.objects)
    assert not any(p.applied or p.validated for p in rec.parameters)


def test_distinct_balls_keep_their_own_parameters():
    result = real_result()
    analysis = semantic_balls(result)
    analysis.parameters = [hybrid.VisionParameter(key='mass_kg',value=value,unit='kg',text=f'{value}kg',
                                                object_index=i,confidence=.99) for i,value in enumerate((4,2))]
    rec = hybrid.fuse(result,analysis).recognition
    assert [obj.body_id for obj in rec.objects] == ['ball-1','ball-2']
    assert [(p.target,p.value,p.validated,p.applied) for p in rec.parameters] == [
        ('ball-1',4,True,True),('ball-2',2,True,False)]


def test_real_block_fixture_still_has_one_unique_ink_supported_quadrilateral():
    path = FIXTURES / 'real/IMG_9142.jpg'
    result = recognize(cap.capture(cap.load_image(path.read_bytes())),path.stem)
    analysis = hybrid.VisionAnalysis(objects=[
        {'kind':'ball','bbox':[367,278,431,379],'confidence':.95},
        {'kind':'block','bbox':[617,641,662,727],'confidence':.95}],
        ramp=None,ground=None,summary='one ball and one block')
    rec = hybrid.fuse(result,analysis).recognition
    assert len(rec.blocks) == 1
    assert rec.objects[1].body_id == 'block-1' and rec.objects[1].geometry_verified
