"""Answer a learner question: LLM if available, else template; always number-checked."""

import re

from app import store
from app.contracts import AskResponse, Level
from app.ta import followups, llm, numcheck
from app.ta.context import build_context, context_numbers, to_json
from app.ta.prompts import system_prompt, user_message
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


def ask(session: store.Session, question: str, level: Level) -> AskResponse:
    session.profile.level = level
    session.log("question", question=question, level=level)
    context = build_context(session, level)
    text = None
    if session.runs:
        text = llm.chat([
            {"role": "system", "content": system_prompt(level)},
            {"role": "user", "content": user_message(to_json(context), question)},
        ])
    source = "llm" if text else "template"
    if not text:
        text = template_answer(session, question, level)
    flags = numcheck.check(text, context_numbers(context, session.runs))
    response = AskResponse(
        answer_md=text,
        based_on=based_on(session, text),
        flags=flags,
        followups=followups.suggest(session),
        source=source,
    )
    session.log("answer", source=source, based_on=response.based_on, flags=flags)
    return response
