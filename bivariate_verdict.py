"""
bivariate_verdict.py
=======================
Parser for the bivariate spatial correlation answer format:
    FINAL ANSWER: POSITIVE (Lee's L = <value>, p = <value>)
    FINAL ANSWER: NEGATIVE (Lee's L = <value>, p = <value>)
    FINAL ANSWER: NOT SIGNIFICANT (Lee's L = <value>, p = <value>)

Fully separate from casual_verdict.py -- does not import from or modify
it in any way, so the existing 74-chunk Moran's I pipeline is completely
unaffected regardless of what happens here. Reuses the SAME proven
self-admission-of-non-computation detection and Python-exception-name
detection logic (built and validated earlier this session), since those
failure modes are generic and apply to any task, not just Moran's I.
"""

import re

NUMBER_PATTERN = r"[-+]?\d*\.?\d+"

# same phrase list validated earlier against real observed fabrication cases
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


def _has_self_admitted_non_computation(full_text):
    if not isinstance(full_text, str):
        return False
    lower = full_text.lower()
    if any(phrase in lower for phrase in SELF_ADMITTED_NON_COMPUTATION_PHRASES):
        return True
    if PYTHON_EXCEPTION_PATTERN.search(full_text):
        return True
    return False


def parse_bivariate_final_answer(final_text):
    """Returns (predicted_direction: str|None, verdict_detail: str, evidence_text: str).
    predicted_direction is one of 'positive', 'negative', 'not_significant', or None
    if no valid answer could be parsed.

    NOTE: agent_loop() already strips everything before "FINAL ANSWER:" and
    returns only the text AFTER it as final_text -- so this function must
    handle BOTH cases: the marker already stripped (the normal case, since
    that's what agent_loop() hands us) and the marker still present (in
    case this is ever called on raw, unstripped model output elsewhere).
    Searching for the marker and falling back to the whole text mirrors
    the exact same defensive pattern already proven in casual_verdict.py.
    """
    if not isinstance(final_text, str) or not final_text.strip():
        return None, "(empty response)", ""

    match = re.search(r"final answer:", final_text, re.IGNORECASE)
    answer_line = final_text[match.end():].strip() if match else final_text.strip()
    answer_line = answer_line.splitlines()[0].strip() if answer_line else ""
    answer_upper = answer_line.upper()

    if "NOT SIGNIFICANT" in answer_upper:
        predicted = "not_significant"
    elif "POSITIVE" in answer_upper:
        predicted = "positive"
    elif "NEGATIVE" in answer_upper:
        predicted = "negative"
    else:
        predicted = None

    return predicted, final_text.strip()[:500], answer_line[:1000]


def parse_bivariate_with_citation(final_text):
    """Returns (predicted_direction, verdict_detail, evidence_text,
    has_citation: bool, cited_numbers: list[float]).
    Same evidence-grounding logic as casual_verdict.py's citation checker:
    a real number must be present in the answer line, AND the model must
    not have admitted elsewhere that the number isn't real."""
    predicted, verdict_detail, evidence_text = parse_bivariate_final_answer(final_text)
    cited_numbers = [float(x) for x in re.findall(NUMBER_PATTERN, evidence_text)] if evidence_text else []

    if _has_self_admitted_non_computation(final_text):
        return predicted, verdict_detail, evidence_text, False, []

    has_citation = len(cited_numbers) > 0
    return predicted, verdict_detail, evidence_text, has_citation, cited_numbers
