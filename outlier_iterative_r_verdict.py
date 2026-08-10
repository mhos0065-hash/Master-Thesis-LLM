"""
outlier_iterative_r_verdict.py
=================================
Parser for the iterative-r outlier answer format:
    FINAL ANSWER: OUTLIER - HIGH (z = <value>)
    FINAL ANSWER: OUTLIER - LOW (z = <value>)
    FINAL ANSWER: NOT AN OUTLIER (z = <value>)

Fully separate from casual_verdict.py and the other task-specific
parsers -- does not import from or modify any of them.
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


def _has_self_admitted_non_computation(full_text):
    if not isinstance(full_text, str):
        return False
    lower = full_text.lower()
    if any(phrase in lower for phrase in SELF_ADMITTED_NON_COMPUTATION_PHRASES):
        return True
    if PYTHON_EXCEPTION_PATTERN.search(full_text):
        return True
    return False


def parse_outlierr_final_answer(final_text):
    """Returns (predicted_direction: str|None, verdict_detail: str, evidence_text: str).
    predicted_direction is one of 'outlier_high', 'outlier_low', 'not_outlier', or None."""
    if not isinstance(final_text, str) or not final_text.strip():
        return None, "(empty response)", ""

    match = re.search(r"final answer:", final_text, re.IGNORECASE)
    answer_line = final_text[match.end():].strip() if match else final_text.strip()
    answer_line = answer_line.splitlines()[0].strip() if answer_line else ""
    answer_upper = answer_line.upper()

    if "NOT AN OUTLIER" in answer_upper:
        predicted = "not_outlier"
    elif "OUTLIER" in answer_upper and "HIGH" in answer_upper:
        predicted = "outlier_high"
    elif "OUTLIER" in answer_upper and "LOW" in answer_upper:
        predicted = "outlier_low"
    else:
        predicted = None

    return predicted, final_text.strip()[:500], answer_line[:1000]


def parse_outlierr_with_citation(final_text):
    """Returns (predicted_direction, verdict_detail, evidence_text,
    has_citation: bool, cited_numbers: list[float])."""
    predicted, verdict_detail, evidence_text = parse_outlierr_final_answer(final_text)

    if _has_self_admitted_non_computation(final_text):
        return predicted, verdict_detail, evidence_text, False, []

    cited_numbers = [float(x) for x in re.findall(NUMBER_PATTERN, evidence_text)] if evidence_text else []
    has_citation = len(cited_numbers) > 0
    return predicted, verdict_detail, evidence_text, has_citation, cited_numbers
