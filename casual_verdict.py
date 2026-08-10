"""
casual_verdict.py
====================
Parses the casual-style FINAL ANSWER format used by the spatial_correlation
prompts (plain text, not JSON):
  - Global: "FINAL ANSWER: CLUSTERED" / "FINAL ANSWER: NOT CLUSTERED"
  - Local:  "FINAL ANSWER: BoroughName1 (HH), BoroughName2 (LL), ..." or "FINAL ANSWER: none"

Reuses the exact extraction functions already validated in
score_moran_boroughs.py (extract_boroughs_from_text) rather than
duplicating that logic -- so a chunk scored during a live run and the same
chunk scored later by score_moran_boroughs.py use identical parsing.
"""

import re
from score_moran_boroughs import extract_boroughs_from_text


def parse_casual_final_answer(final_text, spatial_granularity):
    """Returns (predicted_bool_or_None, verdict_detail, evidence_text).
    predicted is None only if the text is completely unparseable (counts
    as an incorrect/ambiguous prediction downstream, same convention as
    the JSON path's None verdict)."""
    if not isinstance(final_text, str) or not final_text.strip():
        return None, "(empty response)", ""

    match = re.search(r"final answer:", final_text, re.IGNORECASE)
    answer_line = final_text[match.end():].strip() if match else final_text.strip()
    # keep just the first line after the marker -- that's the structured part
    answer_line = answer_line.splitlines()[0].strip() if answer_line else ""

    if spatial_granularity == "global":
        text_upper = answer_line.upper()
        if "NOT CLUSTERED" in text_upper or "NOT_CLUSTERED" in text_upper:
            predicted = False
        elif "CLUSTERED" in text_upper:
            predicted = True
        else:
            predicted = None
        verdict_detail = final_text.strip()[:500]
        evidence_text = answer_line[:1000]
        return predicted, verdict_detail, evidence_text

    else:  # local
        if re.fullmatch(r"none", answer_line.strip(), re.IGNORECASE):
            predicted = False
        else:
            found = extract_boroughs_from_text(final_text)
            predicted = len(found) > 0 if answer_line else None
        verdict_detail = final_text.strip()[:500]
        evidence_text = answer_line[:1000]
        return predicted, verdict_detail, evidence_text


NUMBER_PATTERN = r"[-+]?\d*\.?\d+"

# Phrases that mean "this number is not a real computed result" -- pulled
# directly from real observed cases where a model admitted failure in its
# own reasoning, then still cited a number anyway (e.g. "I cannot provide
# a computed number... FINAL ANSWER: NOT CLUSTERED (Moran's I = 0.05...)").
# If ANY of these appear anywhere in the full response, the citation is
# rejected regardless of whether a number-shaped token is present -- the
# model's own words are the strongest signal that the number isn't real.
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

# Common Python exception class names -- if any of these appear in the
# response, it means real code genuinely crashed. A number sitting next to
# an exception name (e.g. "IndexError: index 2 is out of bounds") is very
# likely a stray digit from the traceback itself, not a real statistic --
# this is a structural check (works for ANY exception type) rather than
# trying to anticipate every way a model might phrase "my code failed".
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


def parse_casual_with_citation(final_text, spatial_granularity):
    """HYBRID format: same casual CLUSTERED/HH-LL wording, but the model is
    additionally required to cite at least one real number in the FINAL
    ANSWER line itself, e.g.:
        FINAL ANSWER: CLUSTERED (Moran's I = 0.17, p = 0.006)
        FINAL ANSWER: Camden (HH, I=1.04), Westminster (HH, I=1.34)
    Returns (predicted_bool_or_None, verdict_detail, evidence_text,
    has_citation: bool, cited_numbers: list[float]). If no number is found
    in the answer line, OR if the model's own text anywhere admits the
    number isn't real (e.g. "could not be computed", "SyntaxError", "N/A"),
    has_citation=False -- a model telling you its own number is fake must
    never count as valid grounded evidence, even if a number-shaped token
    is technically present.
    """
    predicted, verdict_detail, evidence_text = parse_casual_final_answer(final_text, spatial_granularity)
    cited_numbers = [float(x) for x in re.findall(NUMBER_PATTERN, evidence_text)] if evidence_text else []

    if _has_self_admitted_non_computation(final_text):
        return predicted, verdict_detail, evidence_text, False, []

    has_citation = len(cited_numbers) > 0
    return predicted, verdict_detail, evidence_text, has_citation, cited_numbers
