"""Flag numbers in a TA answer that don't come from the context (P3.4).

Extracts decimals with their unit ("1.4 s", "2.46×", "30°", "46%"), plain fractions ("5/7")
and LaTeX fractions (\\frac{5}{7}). A number is accepted if it matches a context number within
2%, is consistent with rounding one (0.4 for 0.395), or is a small textbook constant. "Run N"
labels are not numbers. Flags are returned as written so the UI can mark them; nothing is
regenerated.
"""

import re

REL_TOL = 0.02
WHITELIST = {0.0, 0.5, 9.81, 1.62, 3.71, 90.0, 180.0, 360.0, 100.0,
             5 / 7, 7 / 5, 2 / 3, 2 / 7, 1 / 2, 0.4, 2 / 5, 10 / 7}
SMALL_INTS = set(range(0, 11))

RUN_REF = re.compile(r"\bRuns?\s+\d+(?:\s*(?:,|and|&|vs\.?|–|-|to)\s*(?:Run\s+)?\d+)*", re.I)
LATEX_FRAC = re.compile(r"\\[dt]?frac\{\s*(\d+(?:\.\d+)?)\s*\}\{\s*(\d+(?:\.\d+)?)\s*\}")
PLAIN_FRAC = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?![\w.])")
NUMBER = re.compile(
    r"(?<![\w.\\])[-−]?\d+(?:\.\d+)?"
    r"(?:\s*(?:×|x(?![a-z])|%|°|m/s²|m/s\^2|m/s|kg|s\b|m\b))?",
)
LATEX_POWER = re.compile(r"\^\{?\d+\}?")


def _value(token: str) -> tuple[float, int]:
    num = re.match(r"[-−]?\d+(?:\.\d+)?", token).group().replace("−", "-")
    decimals = len(num.split(".")[1]) if "." in num else 0
    return float(num), decimals


def _accepted(value: float, decimals: int, allowed: list[float]) -> bool:
    v = abs(value)
    if v in SMALL_INTS and decimals == 0:
        return True
    if any(abs(v - w) < 1e-9 for w in WHITELIST):
        return True
    half_step = 0.5 * 10 ** (-decimals) + 1e-9
    for c in allowed:
        c = abs(c)
        if c == 0:
            continue
        if abs(v - c) <= REL_TOL * c or abs(v - c) <= half_step:
            return True
    return False


def check(answer: str, allowed: list[float]) -> list[str]:
    text = RUN_REF.sub(" ", answer)
    text = LATEX_POWER.sub(" ", text)
    flags: list[str] = []

    def flag(token: str) -> None:
        token = token.strip()
        if token not in flags:
            flags.append(token)

    for pattern in (LATEX_FRAC, PLAIN_FRAC):
        for m in pattern.finditer(text):
            a, b = float(m.group(1)), float(m.group(2))
            if b and not _accepted(a / b, 3, allowed) and not (
                    _accepted(a, 0, allowed) and _accepted(b, 0, allowed)):
                flag(m.group(0))
        text = pattern.sub(" ", text)

    for m in NUMBER.finditer(text):
        value, decimals = _value(m.group(0))
        if not _accepted(value, decimals, allowed):
            flag(m.group(0))
    return flags
