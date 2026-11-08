"""Answer a learner question: LLM if available, else template; always number-checked."""

import re

from app import store
from app.contracts import AskResponse, Level
from app.ta import followups, llm, numcheck
from app.ta.context import build_context, context_numbers, to_json
from app.ta.prompts import FOLLOWUP_TASK, system_prompt, user_message
from app.ta.templates import template_answer


def based_on(session: store.Session, answer: str) -> list[str]:
    """Runs cited as "Run N" in the answer; else the current run."""
    by_index = {r.index: r.run_id for r in session.runs}
    cited = []
    for m in re.finditer(r"\bRuns?\s+(\d+(?:\s*(?:,|and|&|→|->|vs\.?)\s*(?:Run\s+)?\d+)*)", answer):
        for n in re.findall(r"\d+", m.group(1)):
            rid = by_index.get(int(n))
            if rid and rid not in cited:
                cited.append(rid)
    if not cited and session.runs:
        cited = [session.runs[-1].run_id]
    return cited


FOLLOWUP_LINE = re.compile(r"^\s*\**FOLLOW-?UPS?\**\s*:\s*(.+?)\s*$", re.I | re.M)

# questions the offline templates can answer, in the order they are offered
POOL = [
    "How long does it take to reach the bottom?",
    "Does mass matter?",
    "What if friction was higher?",
    "What happens if the ramp is steeper?",
    "What if I use the Moon's gravity?",
    "What did you read from my drawing?",
    "Why does the ball speed up?",
]


def split_suggestions(text: str) -> tuple[str, list[str]]:
    """Cut the trailing "FOLLOWUPS: a | b | c" line off the model's answer."""
    match = None
    for match in FOLLOWUP_LINE.finditer(text):
        pass
    if match is None:
        return text, []
    questions = [q.strip(" -•\"'*") for q in match.group(1).split("|")]
    questions = [q if q.endswith("?") else q + "?" for q in questions if 3 <= len(q) <= 90]
    return (text[:match.start()] + text[match.end():]).strip(), questions[:3]


def suggest_questions(session: store.Session, asked_now: str, wanted: list[str]) -> list[str]:
    """Three distinct questions: the model's own, topped up from POOL, never one already asked."""
    asked = {e.data.get("question", "").strip().lower() for e in session.events if e.kind == "question"}
    asked.add(asked_now.strip().lower())
    out: list[str] = []
    for q in [*wanted, *POOL]:
        if q.strip().lower() not in asked and q.lower() not in (o.lower() for o in out):
            out.append(q)
        if len(out) == 3:
            break
    return out


def ask(session: store.Session, question: str, level: Level) -> AskResponse:
    session.profile.level = level
    session.log("question", question=question, level=level)
    context = build_context(session, level)
    text = None
    if session.runs:
        text = llm.chat([
            {"role": "system", "content": system_prompt(level) + FOLLOWUP_TASK},
            {"role": "user", "content": user_message(to_json(context), question)},
        ])
    text, wanted = split_suggestions(text) if text else (None, [])
    source = "llm" if text else "template"
    if not text:
        text = template_answer(session, question, level)
    flags = numcheck.check(text, context_numbers(context, session.runs))
    response = AskResponse(
        answer_md=text,
        based_on=based_on(session, text),
        flags=flags,
        followups=followups.suggest(session),
        suggestions=suggest_questions(session, question, wanted),
        source=source,
    )
    session.log("answer", source=source, based_on=response.based_on, flags=flags)
    return response
