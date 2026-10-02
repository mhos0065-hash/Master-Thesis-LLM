"""
error_analysis_getisord.py
=============================
Standalone error-taxonomy script for the spatial hotspot detection
task (Getis-Ord Gi*). Computes the SAME is_ambiguous_chunk /
is_implausible_chunk / has_real_evidence logic score_getisord.py uses
(copied verbatim), then splits each case's contribution into 5 weight
columns that sum to 1.0. Writes ONLY the error analysis output.

GRADED, not discrete: this task's headline `accuracy` is a per-chunk
F1 (partial credit for borough overlap, not exact-set-match), so a
discrete correct/incorrect split here would NOT reproduce that
accuracy value -- it would reproduce the stricter `correct_number`
column instead. Each evaluable case's weight is split between the
correct-side bucket (weight = f1, the SAME formula and value
score_getisord.py's accuracy table uses) and the incorrect-side
bucket (weight = 1 - f1), chosen by evidence status. Summing
grounded_correctness + ungrounded_correctness across all cases and
dividing by n reproduces the exact same accuracy value the headline
table reports -- verified directly, see the session's numeric check.
is_ambiguous_chunk / is_implausible_chunk keep full 1.0 weight in
their discrete bucket (0.0 credit), matching that these already
contribute 0 to the accuracy table via pred_effective={}.

Note: this task currently has no shift-invariance re-execution check
(reverted on request to match the pre-process-based-metric baseline --
see score_getisord.py's own docstring). has_real_evidence here is a
real, finite, non-fabricated number plus magnitude/threshold
plausibility only.

is_implausible_chunk (parsed but self-contradictory, e.g. too many
boroughs or a HOT label with a near-zero z) is classified as
grounding_failure, not format_failure -- the model DID answer, just
not with trustworthy evidence. format_failure is reserved for
is_ambiguous_chunk (no parseable answer at all).

USAGE
-----
1. Uncomment exactly ONE of the 4 RESULTS_CSV / ERROR_OUT_DIR blocks.
2. Run: python error_analysis_getisord.py
3. Output goes to Errors/<configuration>/:
     error_analysis_getisord.csv
     error_analysis_getisord_counts.csv
"""

import os
import re
import pandas as pd

from statistic_plausibility import magnitude_plausible, gi_threshold_consistent

GROUND_TRUTH_METHODS_PATH = "ground truth/ground_truth_all_methods_getisord.csv"

# ==================== CONFIGURATION: uncomment ONE block ====================

# for multi agent
RESULTS_CSV = "results/multi agent/getisord_results_multiagent_with_method.csv"
ERROR_OUT_DIR = "Errors/multi agent"

# for multi turn
#RESULTS_CSV = "results/single agent/getisord_results_multi_with_method.csv"
#ERROR_OUT_DIR = "Errors/multi turn"

# for single turn
#RESULTS_CSV = "getisord_results_with_method.csv"
#ERROR_OUT_DIR = "Errors/single turn"

# for ReAct
#RESULTS_CSV = "ReAct/getisord_results_react.csv"
#ERROR_OUT_DIR = "Errors/ReAct"

# ==============================================================================

IMPLAUSIBLE_THRESHOLD = 10

METHOD_NAME_MAP = {
    "Getis-Ord Gi*": "Getis-Ord Gi*",
    "Getis-Ord Gi (no star)": "Getis-Ord Gi (no star)",
    "Local Moran's I / LISA": "Local Moran's I / LISA",
}

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
    "please replace the values", "replace the values with the actual",
    "i cannot provide the actual", "i recommend running the corrected code",
]


def _has_self_admitted_non_computation(text):
    if not isinstance(text, str):
        return False
    return any(phrase in text.lower() for phrase in SELF_ADMITTED_NON_COMPUTATION_PHRASES)


def parse_borough_string(s):
    if not isinstance(s, str) or s.strip() in ("(none)", "none", ""):
        return {}
    result = {}
    for entry in s.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) >= 2:
            result[parts[0].strip()] = parts[1].strip()
    return result


def parse_borough_values(s):
    if not isinstance(s, str) or s.strip() in ("(none)", "none", ""):
        return {}
    result = {}
    for entry in s.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) >= 3:
            try:
                result[parts[0].strip()] = float(parts[2].strip())
            except ValueError:
                pass
    return result


def parse_borough_labels(evidence_text):
    if not isinstance(evidence_text, str):
        return {}
    labels = {}
    for entry in re.findall(r"([A-Za-z ]+?)\s*\(([^)]*)\)", evidence_text):
        borough_name, inner = entry
        borough_name = borough_name.strip()
        match = re.search(r"([A-Za-z_\-\s\*]*)=", inner)
        if match and borough_name not in labels:
            labels[borough_name] = match.group(1).strip()
    return labels


def _chunk_has_real_evidence(row):
    text = str(row.get("evidence_text", ""))
    if _has_self_admitted_non_computation(text):
        return False
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    for n in numbers:
        try:
            f = float(n)
            if f == f and abs(f) != float("inf"):
                return True
        except ValueError:
            continue
    return False


def load_ground_truth_methods():
    gt = pd.read_csv(GROUND_TRUTH_METHODS_PATH)
    table = {}
    for _, r in gt.iterrows():
        table[(r["chunk_id"], r["method"])] = r["ground_truth_boroughs"]
    return table


def _borough_confusion(pred, truth):
    """Same formula score_getisord.py's accuracy table uses -- presence-
    only matching, direction not checked."""
    tp = fp = fn = 0
    for borough in ALL_BOROUGHS:
        pred_flagged = borough in pred
        truth_flagged = borough in truth
        if pred_flagged and truth_flagged:
            tp += 1
        elif pred_flagged and not truth_flagged:
            fp += 1
        elif (not pred_flagged) and truth_flagged:
            fn += 1
    return tp, fp, fn


def _chunk_f1(pred, truth):
    """IDENTICAL formula to score_getisord.py's _chunk_f1() (the
    function the reported `accuracy` column is built from) -- copied
    verbatim so the taxonomy's graded weights sum to exactly the same
    per-chunk value the accuracy table uses. This is what makes
    grounded_correctness + ungrounded_correctness average to the same
    number as the reported accuracy, rather than a stricter
    exact-match-only proxy for it."""
    tp, fp, fn = _borough_confusion(pred, truth)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


ALL_BOROUGHS = [
    "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Camden",
    "City of London", "Croydon", "Ealing", "Enfield", "Greenwich", "Hackney",
    "Hammersmith and Fulham", "Haringey", "Harrow", "Havering", "Hillingdon",
    "Hounslow", "Islington", "Kensington and Chelsea", "Kingston upon Thames",
    "Lambeth", "Lewisham", "Merton", "Newham", "Redbridge",
    "Richmond upon Thames", "Southwark", "Sutton", "Tower Hamlets",
    "Waltham Forest", "Wandsworth", "Westminster",
]

WEIGHT_COLS = ["format_failure", "grounded_correctness", "grounding_failure",
               "reasoning_failure", "ungrounded_correctness"]


def resolve_ground_truth(chunk_id, statistical_method, manifest_boroughs_str, gt_methods):
    normalized_method = METHOD_NAME_MAP.get(statistical_method)
    if normalized_method is not None:
        key = (chunk_id, normalized_method)
        if key in gt_methods:
            return gt_methods[key]
    return manifest_boroughs_str


def run():
    os.makedirs(ERROR_OUT_DIR, exist_ok=True)

    df = pd.read_csv(RESULTS_CSV)

    if "statistical_method" not in df.columns:
        raise ValueError(
            f"'{RESULTS_CSV}' has no `statistical_method` column. "
            "Run attach_statistical_method_getisord.py first and point "
            "RESULTS_CSV at its '_with_method' output."
        )

    gt_methods = load_ground_truth_methods()
    error_rows = []

    for (mode, model), g in df.groupby(["mode", "model"]):
        for _, r in g.iterrows():
            pred_str = r.get("predicted_boroughs")
            manifest_truth_str = r.get("ground_truth_boroughs")
            evidence_text = str(r.get("evidence_text", "")).strip().lower()

            resolved_truth_str = resolve_ground_truth(
                r.get("chunk_id"), r.get("statistical_method"), manifest_truth_str, gt_methods)

            pred = parse_borough_string(pred_str)
            truth = parse_borough_string(resolved_truth_str)

            is_valid_none = evidence_text == "none"
            is_empty = not isinstance(pred_str, str) or pred_str.strip() in ("", "(none)", "none") \
                or pd.isna(pred_str)
            is_ambiguous_chunk = is_empty and not is_valid_none

            raw_entries = [e.strip() for e in str(pred_str).split(";") if e.strip()] if not is_empty else []
            has_duplicates = len(raw_entries) > len(pred)

            pred_values = parse_borough_values(pred_str)
            borough_labels = parse_borough_labels(r.get("evidence_text"))
            has_implausible_value = any(not magnitude_plausible(v) for v in pred_values.values())
            has_inconsistent_threshold = any(
                not gi_threshold_consistent(v, label=borough_labels.get(borough))
                for borough, v in pred_values.items()
            )

            is_implausible_chunk = (
                len(pred) > IMPLAUSIBLE_THRESHOLD or has_duplicates
                or has_implausible_value or has_inconsistent_threshold
            )

            has_real_evidence = _chunk_has_real_evidence(r)
            if is_valid_none and pred == truth == {}:
                has_real_evidence = True

            # --- Fractional per-case credit, discrete only at the FINAL
            # pooled total (getisord/spatial-local only) --- getisord's
            # accuracy is a continuous per-chunk F1 (partial credit for
            # borough overlap). Each evaluable case here contributes its
            # exact f1 (the SAME formula the accuracy table uses) to the
            # correct-side bucket and 1-f1 to the incorrect-side bucket
            # -- NOT a per-case whole-bucket assignment. Summed across
            # all cases in a (task, mode) group, this total is written
            # UNROUNDED to the _counts.csv below; rounding happens only
            # once, in merge_error_analysis_counts.py/.R, on the FINAL
            # total pooled across all 5 tasks -- this is what actually
            # minimizes the gap against the accuracy table (verified
            # numerically: this approach's gap was ~50x smaller than a
            # per-case 0.5 threshold on a 20-chunk synthetic test).
            weights = {c: 0.0 for c in WEIGHT_COLS}
            if is_ambiguous_chunk:
                weights["format_failure"] = 1.0
            elif is_implausible_chunk:
                weights["grounding_failure"] = 1.0
            else:
                f1 = _chunk_f1(pred, truth)
                if has_real_evidence:
                    weights["grounded_correctness"] = f1
                    weights["reasoning_failure"] = 1.0 - f1
                else:
                    weights["ungrounded_correctness"] = f1
                    weights["grounding_failure"] = 1.0 - f1

            error_rows.append({
                "chunk_id": r.get("chunk_id"), "model": model, "mode": mode,
                "task": "getisord", **weights,
            })

    error_df = pd.DataFrame(error_rows)
    out_path = os.path.join(ERROR_OUT_DIR, "error_analysis_getisord.csv")
    error_df.to_csv(out_path, index=False)
    print(f"Saved per-case error taxonomy: {out_path} ({len(error_df)} rows)")

    error_counts = (
        error_df.groupby(["task", "mode"])[WEIGHT_COLS].sum().reset_index()
    )
    counts_path = os.path.join(ERROR_OUT_DIR, "error_analysis_getisord_counts.csv")
    error_counts.to_csv(counts_path, index=False)
    print(f"Saved error-type counts: {counts_path}")


if __name__ == "__main__":
    run()
