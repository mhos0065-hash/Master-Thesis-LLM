"""
statistic_plausibility.py
============================
Shared plausibility checks used across all 5 scorers. A row fails
plausibility if ANY of the following independent checks fail -- they
are NOT combined into one condition, since a real broken case
(statistic=128 million, p=1.0, verdict=not_significant) is internally
CONSISTENT (both the p-value and verdict agree it's not significant)
but has an implausible magnitude -- only the magnitude check catches
it. A hypothetical (verdict=positive, p=0.9) case has a plausible
magnitude but an inconsistent verdict -- only the consistency check
catches that one. Both checks are needed, independently, as separate
gates.

1. Magnitude sanity: |primary_statistic| <= 1000. Deliberately generous
   -- six-plus orders of magnitude above any real value seen across this
   project's actual data (max ~6, e.g. outlier z-scores), so it only
   catches clearly broken computations (e.g. forgetting to normalize by
   variance), never a legitimate alternative statistical method.

2. Bounded-probability check: any value used as a p-value or R-squared
   must be in [0, 1]. Definitional and universal -- true for any valid
   statistical method, not specific to Lee's L or Moran's I.

3. Verdict/significance consistency: a verdict that CLAIMS a real,
   significant effect requires p < 0.05; a verdict that claims NO
   significant effect requires p >= 0.05. Uses the exact alpha=0.05
   threshold that every task's own prompt already specifies. For
   Getis-Ord specifically (threshold-based rather than p-value-based,
   per that task's own prompt explicitly stating |z| > 1.96), a
   HOT/COLD label requires |Gi*| >= 1.96.

All functions return True (plausible / nothing to check) when given
None, so callers can call them unconditionally without first checking
whether a given value exists for this task/row.
"""

MAGNITUDE_BOUND = 1000.0
ALPHA = 0.05
GI_THRESHOLD = 1.96


def _to_float(value):
    try:
        f = float(value)
        return f if f == f and abs(f) != float("inf") else None  # excludes nan/inf
    except (ValueError, TypeError):
        return None


def magnitude_plausible(value, bound=MAGNITUDE_BOUND):
    """True if value is None, non-numeric, or within the sanity bound."""
    f = _to_float(value)
    if f is None:
        return True
    return abs(f) <= bound


def bounded_probability(value, lo=0.0, hi=1.0):
    """True if value is None, non-numeric, or within [lo, hi]."""
    f = _to_float(value)
    if f is None:
        return True
    return lo <= f <= hi


def verdict_p_consistent(claims_significance, p_value, alpha=ALPHA):
    """claims_significance: bool, whether the verdict asserts a real
    effect. p_value: float-like or None. True if p_value is None
    (nothing to check) or the verdict and p-value agree with each
    other under the standard alpha threshold."""
    p = _to_float(p_value)
    if p is None:
        return True
    return (p < alpha) if claims_significance else (p >= alpha)


def gi_threshold_consistent(gi_value, threshold=GI_THRESHOLD, alpha=ALPHA):
    """For Getis-Ord specifically: a borough labeled HOT/COLD must be
    justified by its OWN reported value, under whichever interpretation
    that value's magnitude is consistent with -- since the task's own
    prompt states '|z| > 1.96' conditionally ("IF your method produces a
    z-score"), an alternative, permutation-based method could
    legitimately report a p-value instead (always in [0,1]).
      - Value in [0,1]: treated as a p-value -- HOT/COLD requires p < 0.05,
        same alpha every other task's prompt already uses.
      - Value outside [0,1]: treated as a z-score -- HOT/COLD requires
        |z| >= 1.96, exactly as this task's own prompt states."""
    f = _to_float(gi_value)
    if f is None:
        return True
    if 0.0 <= abs(f) <= 1.0:
        return abs(f) < alpha
    return abs(f) >= threshold
