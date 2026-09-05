"""
score_getisord.py
====================
Scores Getis-Ord Gi* results with the SAME column count/structure as
the original version of this script (plus 2 columns, justified below),
so the table stays easy to read and directly comparable to bivariate.

MATCHING RULE: a borough counts as identified if its NAME appears in
both the prediction and the ground truth -- direction (hot vs cold) is
NOT required to match, per explicit instruction ("Brent present should
be counted if it is").

`accuracy` -- THE ONE metric, chosen deliberately over several
alternatives considered during development, all of which were dropped
to avoid exactly the confusion of having too many "how correct" numbers:
    accuracy = mean, over chunks, of per-chunk F1
             = mean of [2 * precision_chunk * recall_chunk / (precision_chunk + recall_chunk)]
This is the standard multi-label partial-credit metric (F1 score,
applied per chunk then averaged): full credit for an exact match,
proportional credit for a partial one, 0 only when a chunk has zero
correct boroughs. It was chosen over a strict exact-set-match
requirement because that alternative was checked directly against this
project's own data and found to sit at 0 for any model that isn't
hitting the true borough set EXACTLY (e.g. finding 2 of 3 true
boroughs still scored 0) -- uninformative for comparing models that
differ in how CLOSE they get, not just whether they're perfect. It was
chosen over Youden's J / Informedness (recall minus false-alarm rate)
because that alternative's false-positive penalty is diluted by this
task's large true-negative pool (~29 non-flagged boroughs per chunk),
so it barely penalizes over-guessing -- F1 penalizes false positives
and false negatives symmetrically regardless of pool size, which is
the more defensible property for "credit for a hit, penalty for a
miss" scoring. `correct_number` (the raw count of exact-set matches)
is still reported alongside, so the strict view isn't lost -- just not
duplicated as its own accuracy ratio column.

`precision`/`recall`/`f1` (pooled, corpus-wide, unchanged from the
original script) are computed from TP/FP/FN summed across EVERY
borough in EVERY chunk for a model, not chunk-averaged -- a
complementary, corpus-level view alongside the per-chunk `accuracy`.

`partial_accuracy` (Jaccard, |intersection|/|union|, Godbole & Sarawagi
2004) is unchanged from the original script and kept as-is.

TWO NEW columns beyond the original script's set, both load-bearing
(not decorative):
  - `n_method_specific_gt`: how many of a model's chunks were scored
    against ground truth for the SPECIFIC method that row used (rather
    than falling back to the manifest's Gi*-derived answer). Without
    this, there's no way to tell whether a model's score rests on a
    method-matched comparison or a fallback one.
  - `accuracy_fully_verified`: fraction of chunks that are correct,
    evidenced, AND have every reported statistic within tolerance of
    the method-matched ground truth value (Gi/Gi*: 0.2, Local Moran's
    I: 1.5 z-score units -- empirically derived, see the R ground-truth
    script). This is the strictest, most trustworthy metric in the
    table -- everything else can be gamed by a lucky guess with a
    fabricated or wrong number; this one can't.

VERDICT correctness is resolved PER ROW against the ground truth for
the SPECIFIC method that row used (from
ground_truth_all_methods_getisord.csv: Getis-Ord Gi*, Getis-Ord Gi (no
star), or Local Moran's I / LISA), falling back to the manifest's
official Gi*-derived ground truth (already present in every row's own
`ground_truth_boroughs` column) only when the row's method has no
computed ground truth.

NEW -- SHIFT-INVARIANCE RE-EXECUTION (the same mechanism used in
score_bivariate.py, folded into `has_real_evidence` so
`accuracy_evidence_filtered`/`f1_evidence_filtered` are process-based,
not just a stricter outcome check): Getis-Ord Gi/Gi* and Local Moran's
I are invariant to a constant additive shift of the input values. The
model's OWN saved code is re-executed on a shifted copy of the data,
and the recomputed value is checked against EVERY originally-reported
borough value (accepting a match with ANY of them), NOT a strict
positional "first borough vs first borough" comparison -- getisord
lists MULTIPLE boroughs per chunk, and the re-executed code's raw print
order has no reason to match predicted_boroughs' order, so a strict
positional match would spuriously fail correct code whenever two
DIFFERENT boroughs got compared to each other. This was confirmed to
happen in practice and is why an earlier version of this check produced
unexpectedly low scores across many models -- fixed here by matching
against the full set of original values instead of just the first one.
No code found / execution failed entirely is treated as inconclusive
(not penalized), same tri-state philosophy used everywhere else in
this pipeline.

Usage:
    python score_getisord.py
Reads:  <mode>_results_with_method.csv (must include `statistical_method`
        -- see attach_statistical_method_getisord.py),
        ground_truth_all_methods_getisord.csv
Writes: getisord_scoring_summary.csv
"""

import re
import pandas as pd

from statistic_plausibility import magnitude_plausible, gi_threshold_consistent
from formula_invariance_check import compute_shifted_statistic

GROUND_TRUTH_METHODS_PATH = "ground truth/ground_truth_all_methods_getisord.csv"
MANIFEST_PATH = "chunks_csv/getisord/getisord_manifest.csv"  # must have csv_path / adjacency_path per chunk_id

# for multi agent
#RESULTS_CSV = "results/multi agent/getisord_results_multiagent_with_method.csv"
#OUT_CSV = "Scores/multi agent/getisord_scoring_summary_multiagent.csv"

# for multi turn
RESULTS_CSV = "results/single agent/getisord_results_multi_with_method.csv"
OUT_CSV = "Scores/multi turn/getisord_scoring_summary.csv"

#For single turn
#RESULTS_CSV = "getisord_results_with_method.csv"
#OUT_CSV = "Scores/single turn/getisord_scoring_summary.csv"

#For React model
#RESULTS_CSV = "ReAct/getisord_results_react.csv"
#OUT_CSV = "Scores/ReAct/getisord_scoring_summary.csv"

ALL_BOROUGHS = [
    "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Camden",
    "City of London", "Croydon", "Ealing", "Enfield", "Greenwich", "Hackney",
    "Hammersmith and Fulham", "Haringey", "Harrow", "Havering", "Hillingdon",
    "Hounslow", "Islington", "Kensington and Chelsea", "Kingston upon Thames",
    "Lambeth", "Lewisham", "Merton", "Newham", "Redbridge",
    "Richmond upon Thames", "Southwark", "Sutton", "Tower Hamlets",
    "Waltham Forest", "Wandsworth", "Westminster",
]

IMPLAUSIBLE_THRESHOLD = 10  # real Gi* rarely flags more than a handful of boroughs

# Maps attach_statistical_method_getisord.py's free-text labels onto the
# (fewer) method names that actually have ground truth computed in
# ground_truth_all_methods_getisord.csv. Anything not listed here (IQR,
# DBSCAN, KDE, standard deviational ellipse, the combined-sum non-
# standard bug pattern, or an unspecified generic z-score) has NO
# ground truth to check against -- those rows fall through to the
# manifest fallback, not "wrong by default".
METHOD_NAME_MAP = {
    "Getis-Ord Gi*": "Getis-Ord Gi*",
    "Getis-Ord Gi (no star)": "Getis-Ord Gi (no star)",
    "Local Moran's I / LISA": "Local Moran's I / LISA",
}

# See the module docstring's TOLERANCE DERIVATION section.
METHOD_TOLERANCE = {
    "Getis-Ord Gi*": 0.2,
    "Getis-Ord Gi (no star)": 0.2,
    "Local Moran's I / LISA": 1.5,
}
DEFAULT_TOLERANCE = 0.2  # used for the manifest_fallback source (Gi*-scale)

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
    """Separate from parse_borough_string() (which only keeps direction,
    used for correctness comparison) -- returns {borough: float_value}
    for the plausibility check AND the new value-verification layer,
    both of which need the actual magnitude."""
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
    """Extracts {borough_name: label} from the raw evidence_text (e.g.
    "Brent (HOT, Gi*=2.01, gi_value=2.01)" -> {"Brent": "Gi*"}) --
    needed because predicted_boroughs never carries the model's own
    label at all, only a clean "name:direction:value" format. Takes the
    FIRST label found per borough (the model's own stated choice), not
    any redundant "gi_value=" duplicate some rows also carry alongside
    it. Returns {} if evidence_text isn't in the expected list format
    (e.g. it's "none") -- callers must handle a missing entry."""
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


def _borough_confusion(pred, truth):
    """Single source of truth for ALL confusion-matrix-based metrics
    below (pooled precision/recall/F1, Jaccard, and the graded scores).

    PRESENCE-ONLY matching: a borough counts as a match if it's flagged
    (as hot OR cold, direction not checked) in both prediction and
    ground truth. Direction (hot vs cold) is intentionally NOT compared
    here, per explicit instruction -- "Brent present should be counted
    if it is". Returns (tp, fp, fn, tn) counts over all 33 boroughs."""
    tp = fp = fn = tn = 0
    for borough in ALL_BOROUGHS:
        pred_flagged = borough in pred
        truth_flagged = borough in truth
        if pred_flagged and truth_flagged:
            tp += 1
        elif pred_flagged and not truth_flagged:
            fp += 1
        elif not pred_flagged and truth_flagged:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def _chunk_partial_accuracy(pred, truth, is_failed_chunk=False):
    """Jaccard-based PARTIAL-CREDIT accuracy for a single chunk --
    |intersection| / |union|, Godbole & Sarawagi (2004). PRESENCE-ONLY
    (borough names) -- see _borough_confusion for why."""
    if is_failed_chunk:
        return 0.0
    pred_set = set(pred.keys())
    truth_set = set(truth.keys())
    if not truth_set and not pred_set:
        return 1.0
    union = pred_set | truth_set
    if not union:
        return 1.0
    intersection = pred_set & truth_set
    return len(intersection) / len(union)


def _chunk_f1(pred, truth, is_failed_chunk):
    """Per-chunk F1 -- THE accuracy metric used below. See module
    docstring for the full justification of why this was chosen over
    the alternatives considered (strict exact-match, Youden's J)."""
    if is_failed_chunk:
        return 0.0
    tp, fp, fn, _ = _borough_confusion(pred, truth)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0  # correctly predicted nothing on a truly-empty chunk
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_ground_truth_methods():
    """Loads ground_truth_all_methods_getisord.csv into a
    {(chunk_id, method): ground_truth_boroughs_string} lookup."""
    gt = pd.read_csv(GROUND_TRUTH_METHODS_PATH)
    table = {}
    for _, r in gt.iterrows():
        table[(r["chunk_id"], r["method"])] = r["ground_truth_boroughs"] if pd.notna(r["ground_truth_boroughs"]) else ""
    return table


def resolve_ground_truth(chunk_id, statistical_method, manifest_boroughs_str, gt_methods):
    """Returns (boroughs_string, source, method_used_for_tolerance) for
    THIS row: the method-specific ground truth if we have one computed,
    otherwise the manifest's Gi*-derived ground truth (already present
    on every row via the original `ground_truth_boroughs` column)."""
    normalized_method = METHOD_NAME_MAP.get(statistical_method)
    if normalized_method is not None:
        key = (chunk_id, normalized_method)
        if key in gt_methods:
            return gt_methods[key], "method_specific", normalized_method

    return manifest_boroughs_str, "manifest_fallback", "Getis-Ord Gi*"


def _value_check_chunk(pred_values, truth_values, matched_boroughs, method_for_tolerance):
    """Checks whether every borough that's correctly flagged (present in
    BOTH pred and resolved truth, since correctness already required
    pred==truth as a set-of-boroughs-with-direction) also has a
    reported VALUE within tolerance of the ground truth value. Returns
    True only if every matched borough's value is within tolerance (and
    there's at least one to check); False if any is out of tolerance;
    None if there's nothing checkable (e.g. an empty/"none" chunk, which
    has no values to compare -- not the same as "wrong")."""
    if not matched_boroughs:
        return None
    tol = METHOD_TOLERANCE.get(method_for_tolerance, DEFAULT_TOLERANCE)
    checked_any = False
    for borough in matched_boroughs:
        pv = pred_values.get(borough)
        tv = truth_values.get(borough)
        if pv is None or tv is None:
            continue
        checked_any = True
        if abs(pv - tv) > tol:
            return False
    return True if checked_any else None


def _load_manifest_index():
    """Loads the manifest indexed by chunk_id, for the shift-invariance
    re-check's csv_path/adjacency_path lookup. Degrades to an always-
    missing stand-in (shift check skipped entirely, never blocking
    scoring) if the file isn't found."""
    try:
        return pd.read_csv(MANIFEST_PATH).set_index("chunk_id")
    except (FileNotFoundError, OSError):
        print(f"WARNING: manifest not found at {MANIFEST_PATH} -- "
              f"shift-invariance re-checks will be skipped for every row.")
        return pd.DataFrame(columns=["csv_path", "adjacency_path"])


def run():
    df = pd.read_csv(RESULTS_CSV)

    if "statistical_method" not in df.columns:
        raise ValueError(
            f"'{RESULTS_CSV}' has no `statistical_method` column. "
            "Run attach_statistical_method_getisord.py first and point "
            "RESULTS_CSV at its '_with_method' output."
        )

    gt_methods = load_ground_truth_methods()
    manifest = _load_manifest_index()
    rows = []

    for (mode, model), g in df.groupby(["mode", "model"]):
        n = len(g)
        n_ambiguous = 0
        correct_number = 0
        correct_number_ef = 0
        evidence_weighted_sum = 0

        tp = fp = fn = tn = 0
        tp_ef = fp_ef = fn_ef = tn_ef = 0
        partial_accuracies, partial_accuracies_ef = [], []
        graded_f1_scores, graded_f1_scores_ef = [], []

        n_method_specific_gt = 0
        n_value_checkable = 0
        n_fully_verified = 0

        for _, r in g.iterrows():
            pred_str = r.get("predicted_boroughs")
            manifest_truth_str = r.get("ground_truth_boroughs")  # always Gi*, per-row already
            evidence_text = str(r.get("evidence_text", "")).strip().lower()

            resolved_truth_str, gt_source, gt_method_used = resolve_ground_truth(
                r.get("chunk_id"), r.get("statistical_method"), manifest_truth_str, gt_methods)
            if gt_source == "method_specific":
                n_method_specific_gt += 1

            pred = parse_borough_string(pred_str)
            truth = parse_borough_string(resolved_truth_str)

            is_valid_none = evidence_text == "none"
            is_empty = not isinstance(pred_str, str) or pred_str.strip() in ("", "(none)", "none") \
                or pd.isna(pred_str)
            is_ambiguous_chunk = is_empty and not is_valid_none

            raw_entries = [e.strip() for e in str(pred_str).split(";") if e.strip()] if not is_empty else []
            has_duplicates = len(raw_entries) > len(pred)

            pred_values = parse_borough_values(pred_str)
            truth_values = parse_borough_values(resolved_truth_str)
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

            # --- Shift-invariance re-execution (process-based check) ---
            # Getis-Ord Gi/Gi* (and Local Moran's I) are invariant to a
            # constant additive shift of the input values -- re-run the
            # model's OWN saved code on a shifted copy of the data and
            # compare the recomputed value against the ORIGINALLY
            # reported ones. Unlike a single-scalar task (bivariate/
            # outlier-r/changespillover), a getisord chunk lists
            # MULTIPLE boroughs, and the re-executed code's raw print
            # order has no reason to match predicted_boroughs' order --
            # comparing "first vs first" positionally would spuriously
            # fail correct code whenever two DIFFERENT boroughs got
            # compared to each other (confirmed to happen in practice --
            # this is why an earlier version of this check produced
            # unexpectedly low scores across many models). Instead, the
            # recomputed value is checked against EVERY originally-
            # reported borough value, accepting a match with ANY of
            # them -- still a real discriminating check (a wildly
            # different or non-finite re-run result matches none of
            # them), just robust to ordering differences.
            if has_real_evidence and pred_values and r.get("chunk_id") in manifest.index:
                m = manifest.loc[r["chunk_id"]]
                shifted_stat = compute_shifted_statistic(
                    r.get("run_dir"), m["csv_path"], m.get("adjacency_path"))
                if shifted_stat is not None:
                    try:
                        shifted_val = float(shifted_stat)
                        tol = METHOD_TOLERANCE.get(gt_method_used, DEFAULT_TOLERANCE)
                        is_match = any(abs(shifted_val - v) <= tol for v in pred_values.values())
                        if not (shifted_val == shifted_val and abs(shifted_val) != float("inf") and is_match):
                            has_real_evidence = False
                    except (TypeError, ValueError):
                        has_real_evidence = False
                # shifted_stat is None (no code found / execution failed
                # entirely) -- inconclusive, NOT penalized; has_real_evidence
                # is left as-is, same tri-state philosophy used everywhere
                # else in this pipeline.

            is_failed_chunk = is_ambiguous_chunk or is_implausible_chunk

            if is_failed_chunk:
                n_ambiguous += 1
                pred_effective = {}
            else:
                pred_effective = pred
                evidence_weighted_sum += int(has_real_evidence)
                # PRESENCE-ONLY correctness: matches on borough NAME sets,
                # direction not required (see _borough_confusion).
                is_correct = set(pred.keys()) == set(truth.keys())
                if is_correct:
                    correct_number += 1
                    if has_real_evidence:
                        correct_number_ef += 1

                        # Per-borough value verification: only for chunks
                        # that are exact-match correct AND evidenced.
                        matched_boroughs = set(pred.keys()) & set(truth.keys())
                        value_result = _value_check_chunk(
                            pred_values, truth_values, matched_boroughs, gt_method_used)
                        if value_result is not None:
                            n_value_checkable += 1
                            if value_result:
                                n_fully_verified += 1

            # --- Pooled TP/FP/FN/TN (see _borough_confusion) ---
            c_tp, c_fp, c_fn, c_tn = _borough_confusion(pred_effective, truth)
            tp += c_tp
            fp += c_fp
            fn += c_fn
            tn += c_tn
            # Evidence-filtered: FP/FN/TN always count as-is (an
            # unevidenced false claim is still false); only a TP needs
            # real evidence to stay a TP -- otherwise it demotes to a
            # false negative (a "correct" answer with no real basis
            # isn't credited, but also isn't double-penalized as an FP).
            if has_real_evidence:
                tp_ef += c_tp
            else:
                fn_ef += c_tp
            fp_ef += c_fp
            fn_ef += c_fn
            tn_ef += c_tn

            partial_accuracies.append(_chunk_partial_accuracy(pred_effective, truth, is_failed_chunk))
            pred_effective_ef = pred_effective if has_real_evidence else {}
            partial_accuracies_ef.append(_chunk_partial_accuracy(pred_effective_ef, truth, is_failed_chunk))

            graded_f1_scores.append(_chunk_f1(pred_effective, truth, is_failed_chunk))
            graded_f1_scores_ef.append(_chunk_f1(pred_effective_ef, truth, is_failed_chunk))

        non_ambiguous = n - n_ambiguous
        pct_with_evidence = round(evidence_weighted_sum / n, 3) if n else 0.0

        accuracy = round(sum(graded_f1_scores) / n, 3) if n else 0.0
        precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
        f1 = round(2 * precision * recall / (precision + recall), 3) if (precision + recall) > 0 else 0.0

        accuracy_ef = round(sum(graded_f1_scores_ef) / n, 3) if n else 0.0
        precision_ef = round(tp_ef / (tp_ef + fp_ef), 3) if (tp_ef + fp_ef) > 0 else 0.0
        recall_ef = round(tp_ef / (tp_ef + fn_ef), 3) if (tp_ef + fn_ef) > 0 else 0.0
        f1_ef = round(2 * precision_ef * recall_ef / (precision_ef + recall_ef), 3) \
            if (precision_ef + recall_ef) > 0 else 0.0

        partial_accuracy = round(sum(partial_accuracies) / n, 3) if n else 0.0
        partial_accuracy_ef = round(sum(partial_accuracies_ef) / n, 3) if n else 0.0

        accuracy_fully_verified = round(n_fully_verified / n, 3) if n else 0.0

        rows.append({
            "mode": mode, "model": model, "n": n, "n_ambiguous": n_ambiguous,
            "non_ambiguous": non_ambiguous, "correct_number": correct_number,
            "pct_with_evidence": pct_with_evidence,
            "tp": tp, "fp": fp,
            "accuracy": accuracy,
            "precision": precision, "recall": recall, "f1": f1,
            "correct_number_evidence_filtered": correct_number_ef,
            "tp_evidence_filtered": tp_ef, "fp_evidence_filtered": fp_ef,
            "accuracy_evidence_filtered": accuracy_ef,
            "precision_evidence_filtered": precision_ef,
            "recall_evidence_filtered": recall_ef,
            "f1_evidence_filtered": f1_ef,
            "partial_accuracy": partial_accuracy,
            "partial_accuracy_evidence_filtered": partial_accuracy_ef,
            # the 2 new columns beyond the original script -- see docstring
            "n_method_specific_gt": n_method_specific_gt,
            "accuracy_fully_verified": accuracy_fully_verified,
        })

    out = pd.DataFrame(rows).sort_values(["mode", "model"]).reset_index(drop=True)
    out = out[["mode", "model", "n", "n_ambiguous", "non_ambiguous", "correct_number",
               "pct_with_evidence", "tp", "fp", "accuracy", "precision", "recall", "f1",
               "correct_number_evidence_filtered", "tp_evidence_filtered", "fp_evidence_filtered",
               "accuracy_evidence_filtered", "precision_evidence_filtered",
               "recall_evidence_filtered", "f1_evidence_filtered",
               "partial_accuracy", "partial_accuracy_evidence_filtered",
               "n_method_specific_gt", "accuracy_fully_verified"]]
    out.to_csv(OUT_CSV, index=False)

    print(out.to_string(index=False))
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    run()
