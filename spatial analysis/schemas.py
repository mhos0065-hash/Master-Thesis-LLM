"""
schemas.py
===========
Pydantic-validated structure for the final verdict every agent (single- or
multi-agent Reporter) must produce. This directly implements two things
your proposal specifies that plain FOUND/NOT FOUND text-matching did not:

  - Section 4.2: "Pydantic-validated JSON governing inter-agent communication"
  - Section 4.3: "A result is correct only if it matches human-derived
    ground truth AND cites computed data points as evidence."

`evidence` is a required, non-empty list -- a verdict with zero evidence
items fails validation, which is the mechanism that actually enforces the
"cites computed data points" requirement, rather than trusting the model's
prose to happen to mention a number.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Union
import json
import re


class Evidence(BaseModel):
    description: str = Field(..., min_length=1,
        description="What was computed, e.g. 'modified z-score for Site X'")
    value: Union[float, int, str] = Field(...,
        description="The actual computed number (or short computed string)")


class AgentVerdict(BaseModel):
    insight_present: bool
    verdict_detail: str = Field(..., min_length=1)
    evidence: List[Evidence] = Field(..., min_length=1,
        description="At least one computed value must be cited -- this is "
                     "what makes the verdict 'grounded' rather than asserted.")

    @field_validator("evidence")
    @classmethod
    def evidence_must_be_computed_not_guessed(cls, v):
        if len(v) == 0:
            raise ValueError("evidence list is empty -- a verdict must cite "
                              "at least one computed value (Section 4.3).")
        return v


JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

SCHEMA_INSTRUCTIONS = """
When you give your FINAL ANSWER, it must be a single JSON object (nothing
else after it) matching this exact shape:

{
  "insight_present": true or false,
  "verdict_detail": "one or two sentence plain-English summary of your finding",
  "evidence": [
    {"description": "what you computed", "value": <the actual number>},
    ... (at least one entry -- you MUST cite a real number you computed with code)
  ]
}

Do not include any text after the closing brace. Example:

FINAL ANSWER:
{"insight_present": true, "verdict_detail": "Site X is a statistical outlier.",
 "evidence": [{"description": "modified z-score for Site X", "value": 4.12},
              {"description": "threshold used", "value": 3.0}]}
"""


def parse_verdict(final_answer_text):
    """Extract and validate a JSON verdict from the model's FINAL ANSWER text.
    Returns (AgentVerdict | None, error_message | None). None verdict means
    the reply either wasn't valid JSON or failed schema validation -- both
    cases should score as an ambiguous/incorrect prediction, never silently
    coerced into a guess."""
    match = JSON_BLOCK_RE.search(final_answer_text)
    if not match:
        return None, "no JSON object found in FINAL ANSWER"
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    try:
        verdict = AgentVerdict.model_validate(raw)
        return verdict, None
    except Exception as e:
        return None, f"schema validation failed: {e}"
