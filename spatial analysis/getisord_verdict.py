"""
getisord_verdict.py
======================
Parser for the Getis-Ord answer format:
    FINAL ANSWER: BoroughName1 (HOT, Gi*=<value>), BoroughName2 (COLD, Gi*=<value>), ...
    FINAL ANSWER: none

Fully separate from casual_verdict.py -- does not import from or modify
it in any way. Reuses the same proven self-admission/exception detection
logic validated earlier this session.
"""

import re

NUMBER_PATTERN = r"[-+]?\d*\.?\d+"

SELF_ADMITTED_NON_COMPUTATION_PHRASES = [
    "not computed", "could not be computed", "unable to compute", "impossible to compute",
    "no numerical results", "did not reach a valid", "no valid final answer",
    "not available", "cannot provide a computed number", "no computed",
    "did not execute", "not execute correctly", "syntaxerror", "not provided",
    "could not compute", "n/a", "[not computed]", "placeholder value",
    "not actual computation", "can't report a valid", "cannot report a valid",
    "failed to converge", "insufficient data", "insufficient information",
    "could not be completed", "cannot be determined", "don't include actual",
    "do not include actual", "numerical issue that prevented",
]

PYTHON_EXCEPTION_PATTERN = re.compile(
    r"\b(IndexError|KeyError|ValueError|TypeError|AttributeError|FileNotFoundError|"
    r"ModuleNotFoundError|ZeroDivisionError|NameError|ImportError|RuntimeError)\b"
)

BOROUGH_NAMES = [
    "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Camden",
    "City of London", "Croydon", "Ealing", "Enfield", "Greenwich", "Hackney",
    "Hammersmith and Fulham", "Haringey", "Harrow", "Havering", "Hillingdon",
    "Hounslow", "Islington", "Kensington and Chelsea", "Kingston upon Thames",
    "Lambeth", "Lewisham", "Merton", "Newham", "Redbridge",
    "Richmond upon Thames", "Southwark", "Sutton", "Tower Hamlets",
    "Waltham Forest", "Wandsworth", "Westminster",
]
# match longer names first so "Kingston upon Thames" isn't shadowed by a shorter partial match
BOROUGH_NAMES_SORTED = sorted(BOROUGH_NAMES, key=len, reverse=True)


def _has_self_admitted_non_computation(full_text):
    if not isinstance(full_text, str):
        return False
    lower = full_text.lower()
    if any(phrase in lower for phrase in SELF_ADMITTED_NON_COMPUTATION_PHRASES):
        return True
    if PYTHON_EXCEPTION_PATTERN.search(full_text):
        return True
    return False


def extract_hot_cold_boroughs(text):
    """Returns dict {borough_name: (direction, z_value_or_None)} for every
    borough mentioned with a HOT or COLD label immediately following it in
    parentheses -- e.g. "Camden (HOT, Gi*=2.1)". Requires the label to be
    tightly bound to the borough name (not just present somewhere nearby),
    since a loose window can otherwise bleed into the NEXT borough's own
    label when two boroughs are mentioned close together, e.g.
    "Kingston upon Thames (COLD, ...), Richmond upon Thames (HOT, ...)"
    -- a wide window around "Kingston upon Thames" would incorrectly also
    see the word HOT from Richmond's mention just after it.
    """
    if not isinstance(text, str):
        return {}
    found = {}
    for borough in BOROUGH_NAMES_SORTED:
        pattern = re.escape(borough) + r"\s*\(\s*(HOT|COLD)\b[^)]*"
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        direction = match.group(1).lower()
        segment = match.group(0)
        # matches ANY label word(s) before '=', not just literal "Gi*" -- the
        # manifest may ask for "z-score", "Gi*", or another statistic name
        # depending on which method the model chose; the label itself doesn't
        # matter, only that a real number follows an '=' sign
        num_match = re.search(r"[A-Za-z*\-\s]+=\s*(" + NUMBER_PATTERN + r")", segment)
        z_value = float(num_match.group(1)) if num_match else None
        found[borough] = (direction, z_value)
    return found


def parse_getisord_final_answer(final_text):
    """Returns (predicted_boroughs: dict|None, verdict_detail: str, evidence_text: str).
    predicted_boroughs maps borough -> (direction, z_value). Empty dict {} means
    the model explicitly said 'none'. None means unparseable."""
    if not isinstance(final_text, str) or not final_text.strip():
        return None, "(empty response)", ""

    match = re.search(r"final answer:", final_text, re.IGNORECASE)
    answer_line = final_text[match.end():].strip() if match else final_text.strip()
    answer_line = answer_line.splitlines()[0].strip() if answer_line else ""

    if re.fullmatch(r"none", answer_line.strip(), re.IGNORECASE):
        return {}, final_text.strip()[:500], answer_line[:1000]

    found = extract_hot_cold_boroughs(answer_line)
    predicted = found if found else None
    return predicted, final_text.strip()[:500], answer_line[:1000]


def parse_getisord_with_citation(final_text):
    """Returns (predicted_boroughs, verdict_detail, evidence_text,
    has_citation: bool, cited_numbers: list[float])."""
    predicted, verdict_detail, evidence_text = parse_getisord_final_answer(final_text)

    if _has_self_admitted_non_computation(final_text):
        return predicted, verdict_detail, evidence_text, False, []

    cited_numbers = [float(x) for x in re.findall(NUMBER_PATTERN, evidence_text)] if evidence_text else []
    has_citation = len(cited_numbers) > 0
    return predicted, verdict_detail, evidence_text, has_citation, cited_numbers
