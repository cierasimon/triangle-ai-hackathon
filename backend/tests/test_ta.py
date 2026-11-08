"""Phase 3: the TA. Template mode must pass everything with no key; the LLM is mocked except
in the opt-in `pytest -m live_llm` test."""

import json
import random
import re

import httpx
import pytest

from app.ta import llm
from tests.conftest import FIXTURES

PHOTO = FIXTURES / "synthetic" / "paperphysics_v3_00000.png"
NO_BALL = FIXTURES / "real" / "IMG_9143.jpg"
QUESTIONS = json.loads((FIXTURES.parent / "backend" / "tests" / "ta_questions.json").read_text())
LEVELS = ["beginner", "intermediate", "confident"]


def upload(client, path=PHOTO) -> str:
    with path.open("rb") as f:
        r = client.post("/api/upload", files={"photo": (path.name, f, "image/jpeg")})
    return r.json()["image_id"]


def run(client, image_id, **override) -> dict:
    r = client.post("/api/runs", json={"image_id": image_id, "params_override": override})
    assert r.status_code == 200, r.text
    return r.json()


def ask(client, question, level="beginner") -> dict:
    r = client.post("/api/ta/ask", json={"question": question, "level": level})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture
def lab(client):
    """A session with a few typical changes: mass, Moon, friction that stops it, roll."""
    image_id = upload(client)
    run(client, image_id)
    run(client, image_id, mass_kg=2)
    run(client, image_id, mass_kg=2, gravity=1.62)
    run(client, image_id, mass_kg=2, gravity=1.62, angle_deg=30, friction=0.6)
    run(client, image_id, mass_kg=2, gravity=1.62, angle_deg=30, motion="roll")
    return image_id


# ---- LLM client -------------------------------------------------------------------------


def test_free_only_guard():
    assert llm.check_free("nvidia/nemotron-3.5-super-vl:free")
    assert llm.check_free("openrouter/free")
    for paid in ("openai/gpt-5", "anthropic/claude-opus-5-5", "openrouter/auto", "x:freebie"):
        with pytest.raises(llm.PaidModelError):
            llm.check_free(paid)


@pytest.fixture
def keyed(monkeypatch):
    """LLM enabled with a fake key; requests go to a mock transport, never the network."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "primary/model:free")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "openrouter/free")
    calls = []

    def use(handler):
        def wrapped(request: httpx.Request):
            body = json.loads(request.content)
            calls.append(body)
            return handler(body)
        monkeypatch.setattr(llm, "transport", httpx.MockTransport(wrapped))
        return calls
    return use


def reply(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def test_no_key_sends_nothing(monkeypatch):
    monkeypatch.setattr(llm, "transport", httpx.MockTransport(lambda r: pytest.fail("sent")))
    assert llm.chat([{"role": "user", "content": "hi"}]) is None


def test_primary_ok(keyed):
    calls = keyed(lambda body: reply("hello"))
    assert llm.chat([{"role": "user", "content": "hi"}]) == "hello"
    assert [c["model"] for c in calls] == ["primary/model:free"]
    assert calls[0]["max_tokens"] >= 1000


@pytest.mark.parametrize("failure", [
    lambda body: httpx.Response(429),
    lambda body: httpx.Response(503),
    lambda body: reply(None),  # reasoning model used the whole budget
    lambda body: reply("   "),
])
def test_fallback_once_then_none(keyed, failure):
    calls = keyed(lambda body: failure(body) if body["model"] != "openrouter/free" else reply("ok"))
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok"
    calls.clear()
    keyed(failure)
    assert llm.chat([{"role": "user", "content": "hi"}]) is None


def test_timeout_falls_back(keyed):
    def handler(body):
        if body["model"] == "primary/model:free":
            raise httpx.ReadTimeout("slow")
        return reply("fallback")
    keyed(handler)
    assert llm.chat([{"role": "user", "content": "hi"}]) == "fallback"


def test_paid_model_refused_before_sending(keyed, monkeypatch):
    calls = keyed(lambda body: reply("should not happen"))
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-5")
    with pytest.raises(llm.PaidModelError):
        llm.chat([{"role": "user", "content": "hi"}])
    assert calls == []


# ---- template mode ----------------------------------------------------------------------


def test_template_answers_have_no_invented_numbers(client, lab):
    for question in QUESTIONS:
        for level in LEVELS:
            a = ask(client, question, level)
            assert a["source"] == "template"
            assert a["flags"] == [], (question, level, a["answer_md"], a["flags"])
            assert a["based_on"] and len(a["followups"]) == 2
            assert a["answer_md"].strip()


def test_levels_differ(client, lab):
    answers = {lvl: ask(client, "Why didn't doubling the mass change anything?", lvl)["answer_md"]
               for lvl in LEVELS}
    assert len(set(answers.values())) == 3
    assert "$" not in answers["beginner"].split("\n\n")[0]  # no formulas for beginners
    assert "$" in answers["confident"]


def test_answers_cite_runs(client, lab):
    a = ask(client, "What happens on the Moon?", "intermediate")
    assert "Run 2 → Run 3" in a["answer_md"] and "2.46×" in a["answer_md"]
    assert a["based_on"] == ["r2", "r3"]
    mass = ask(client, "Does mass matter?")
    assert "Run 1 → Run 2" in mass["answer_md"] and "same time" in mass["answer_md"]


def test_out_of_scope(client, lab):
    a = ask(client, "What if I attach a spring to the ball?")
    assert "outside what this lab simulates" in a["answer_md"]


def test_explains_recognition_problems(client, monkeypatch):
    # the photo's drawn square is a block now; switch block detection off to get a picture with no body
    monkeypatch.setattr("app.cv.recognize.detect_blocks", lambda *_: [])
    image_id = upload(client, NO_BALL)
    run(client, image_id)
    a = ask(client, "What did you read from my drawing?")
    assert "no ball" in a["answer_md"] and "To redraw" in a["answer_md"]
    assert "Assumed" in a["answer_md"] and a["flags"] == []


def test_session_log_and_profile(client, lab):
    ask(client, "Does mass matter?", "confident")
    state = client.get("/api/session").json()
    kinds = [e["kind"] for e in state["events"]]
    assert kinds.count("change") == 4 and kinds[-2:] == ["question", "answer"]
    seen = state["profile"]["concepts_seen"]
    assert seen == {"mass_independent": True, "angle_effect": True, "gravity_effect": True,
                    "friction_effect": True, "roll_effect": True}
    assert state["profile"]["level"] == "confident" and state["profile"]["runs"] == 5


def test_followups_come_from_untried_concepts_and_apply(client):
    image_id = upload(client)
    run(client, image_id)
    first = client.get("/api/ta/followups").json()
    assert [f["concept"] for f in first] == ["mass_independent", "gravity_effect"]
    run(client, image_id, **first[1]["change"])  # the learner clicks "Try the Moon"
    after = [f["concept"] for f in client.get("/api/ta/followups").json()]
    assert "gravity_effect" not in after and after[0] == "mass_independent"


# ---- summary ----------------------------------------------------------------------------

STEPS = [{"mass_kg": 2}, {"gravity": 1.62}, {"gravity": 3.71}, {"angle_deg": 15},
         {"angle_deg": 45}, {"friction": 0.1}, {"friction": 0.6}, {"motion": "roll"},
         {"motion": "slide"}, {"gravity": 9.81}]


@pytest.mark.parametrize("seed", range(10))
def test_summary_lists_every_change(client, seed):
    rng = random.Random(seed)
    image_id = upload(client)
    run(client, image_id)
    override: dict = {}
    for _ in range(rng.randint(2, 6)):
        override = {**override, **rng.choice(STEPS)}
        run(client, image_id, **override)
    runs = client.get("/api/session").json()["runs"]
    s = client.post("/api/ta/summary", json={"level": "beginner"}).json()
    assert s["source"] == "template"
    text = s["summary_md"]
    for prev, r in zip(runs, runs[1:]):
        line = next(l for l in text.splitlines() if l.startswith(f"- **Run {r['index']}**"))
        for key, (a, b) in r["changed"].items():
            name = {"angle_deg": "θ", "mass_kg": "mass", "gravity": "g", "friction": "μ"}[key]
            assert re.search(rf"{name} {a:g}[^→]* → {b:g}", line), (key, line)
        if prev["scene"]["params"]["motion"] != r["scene"]["params"]["motion"]:
            assert f"motion {prev['scene']['params']['motion']} →" in line
    assert "Formulas used" in text


def test_llm_summary_is_number_checked(client, lab, monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "Run 1 Run 2 Run 3 Run 4 Run 5: it took 7.77 s.")
    assert client.post("/api/ta/summary").json()["source"] == "template"
    good = " ".join(f"Run {i}" for i in range(1, 6)) + " — mass cancels; the Moon run took longer."
    monkeypatch.setattr(llm, "chat", lambda *a, **k: good)
    s = client.post("/api/ta/summary").json()
    assert s == {"summary_md": good, "source": "llm"}


def test_llm_answer_flags_invented_numbers(client, lab, monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "In Run 3 it took 0.97 s, about 4.4 s on Pluto.")
    a = ask(client, "What happens on the Moon?")
    assert a["source"] == "llm" and a["flags"] == ["4.4 s"] and a["based_on"] == ["r3"]


# ---- quiz -------------------------------------------------------------------------------


def test_quiz(client):
    image_id = upload(client)
    run(client, image_id, mass_kg=3)
    q = client.post("/api/ta/quiz").json()
    assert q["param"] == "gravity" and q["new_value"] == 1.62 and "Run 1" in q["prompt"]
    res = client.post("/api/ta/quiz/answer", json={"quiz_id": q["quiz_id"], "guess": 1.0}).json()
    params = res["run"]["scene"]["params"]
    assert params["gravity"] == 1.62 and params["mass_kg"] == 3  # builds on the learner's setup
    assert res["actual"] == res["run"]["verification"]["simulated_time_s"]
    assert res["error_pct"] == pytest.approx(abs(1.0 - res["actual"]) / res["actual"] * 100)
    assert client.post("/api/ta/quiz/answer",
                       json={"quiz_id": q["quiz_id"], "guess": 1.0}).status_code == 404
    assert "quiz" in [e["kind"] for e in client.get("/api/session").json()["events"]]


# ---- live (opt-in) ----------------------------------------------------------------------


@pytest.mark.live_llm
def test_live_scripted_questions(client, monkeypatch):
    """`pytest -m live_llm`: real free model; no flagged invented numbers. Reads .env."""
    import time

    from app import config
    monkeypatch.delenv("PAPERPHYSICS_NO_DOTENV", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    config._load_dotenv(config.REPO_ROOT / ".env")
    if not config.get_settings().llm_enabled:
        pytest.skip("no OPENROUTER_API_KEY")
    image_id = upload(client)
    run(client, image_id)
    run(client, image_id, mass_kg=2)
    run(client, image_id, mass_kg=2, gravity=1.62)
    report = []
    for i, question in enumerate(QUESTIONS):
        a = ask(client, question, LEVELS[i % 3])
        report.append((question, a["source"], a["flags"]))
        time.sleep(1.5)  # free models are rate-limited
    print("\n".join(f"{s:8} {f} {q}" for q, s, f in report))
    assert any(s == "llm" for _, s, _ in report), "every call fell back to template"
    assert all(not f for _, _, f in report), [r for r in report if r[2]]


def test_new_photo_restarts_runs(client):
    image_id = upload(client)
    run(client, image_id)
    run(client, image_id, gravity=1.62)
    second = upload(client)
    assert run(client, second)["run_id"] == "r1"
    state = client.get("/api/session").json()
    assert [r["index"] for r in state["runs"]] == [1]
    assert [e["kind"] for e in state["events"]].count("upload") == 2
    assert state["profile"]["concepts_seen"]["gravity_effect"]
