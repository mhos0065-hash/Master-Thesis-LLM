"""
score_getisord.py
====================
Scores Getis-Ord Gi* results with the SAME column structure used for
bivariate and the original Moran's I task, so all three tables are
directly comparable side by side.

Structural note: unlike bivariate/Moran's I (one classification per
chunk), Getis-Ord asks for a LIST of boroughs per chunk. To fit the same
column set:
  - tp / fp are BOROUGH-LEVEL, pooled across every (chunk, borough) pair
    -- the standard way to make a variable-length-list task comparable
    to a single-classification task (same approach used earlier this
    session in score_getisord_pooled.py).
  - correct_number is CHUNK-level: a chunk only counts as correct if the
    predicted set of boroughs EXACTLY matches the real significant set.
  - n_ambiguous folds in BOTH genuinely unparseable chunks AND
    implausible ones (e.g. flagging all 33 boroughs, or a computation
    bug producing a wildly-off borough count) -- both are untrustworthy
    for the same reason, so both are excluded from pooling the same way
    and contribute to the "no answer" side of scoring.

Evidence check: same principle as score_bivariate.py -- a chunk only
counts as evidenced if at least one REAL, finite, non-placeholder number
is cited (not nan, not a method name, not a self-admitted "please
replace this" fabrication).

Usage:
    python score_getisord.py
Reads:  getisord_results.csv
Writes: getisord_scoring_summary.csv
"""

import re
import pandas as pd

from statistic_plausibility import magnitude_plausible, gi_threshold_consistent

# for multi agent
RESULTS_CSV = "results/multi agent/getisord_results_multiagent.csv"
OUT_CSV = "Scores/multi agent/getisord_scoring_summary_multiagent.csv"

# for multi turn
#RESULTS_CSV = "results/single agent/getisord_results_multi.csv"
#OUT_CSV = "Scores/multi turn/getisord_scoring_summary.csv"


#For single turn
#RESULTS_CSV = "getisord_results.csv"
#OUT_CSV = "Scores/single turn/getisord_scoring_summary.csv"

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
    for the plausibility check, which needs the actual magnitude."""
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
    """A chunk counts as evidenced only if at least one cited number is a
    real, finite value AND the model didn't admit elsewhere that its
    number isn't real."""
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


def _chunk_partial_accuracy(pred_set, truth_set, is_failed_chunk=False):
    """Jaccard-based PARTIAL-CREDIT accuracy for a single chunk:
    |intersection| / |union|. Standard, citable multi-label metric --
    Godbole & Sarawagi (2004), "Discriminative Methods for
    Multi-labeled Classification", define example-based accuracy for
    multi-label tasks exactly this way. Unlike simple recall
    (TP/real_total), this also penalizes false positives: a model that
    flags every borough would score poorly here, not perfectly, since
    the union grows with every wrong extra guess.

    An ambiguous/implausible chunk (is_failed_chunk=True) always scores
    0.0 here, same principle as correct_number -- it never receives
    credit just because its forced-empty prediction happens to coincide
    with an empty ground truth.

    Genuinely correct empty predictions (both sets empty, and the chunk
    was NOT a failure) score 1.0 -- a real, deliberate "no significant
    boroughs" match. A false "none" against a non-empty truth scores
    0.0, matching the existing empty-truth convention."""
    if is_failed_chunk:
        return 0.0
    if not truth_set and not pred_set:
        return 1.0
    union = pred_set | truth_set
    if not union:
        return 1.0
    intersection = pred_set & truth_set
    return len(intersection) / len(union)


def run():
    df = pd.read_csv(RESULTS_CSV)
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

        for _, r in g.iterrows():
            pred_str = r.get("predicted_boroughs")
            truth_str = r.get("ground_truth_boroughs")
            evidence_text = str(r.get("evidence_text", "")).strip().lower()

            pred = parse_borough_string(pred_str)
            truth = parse_borough_string(truth_str)

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
            # A correct "none" verdict has no number to cite by construction --
            # not because anything is unsupported, but because nothing was
            # flagged. Treat it as evidenced so it isn't systematically
            # excluded from the evidence-filtered metric purely for being
            # correctly negative. Only applies when the prediction is
            # ALSO correct (pred == truth == {}) -- an incorrect "none"
            # (missed real hotspots) still isn't given a pass here.
            if is_valid_none and pred == truth == {}:
                has_real_evidence = True

            if is_ambiguous_chunk or is_implausible_chunk:
                n_ambiguous += 1
                pred_effective = {}
            else:
                pred_effective = pred
                evidence_weighted_sum += int(has_real_evidence)
                if pred == truth:
                    correct_number += 1
                    if has_real_evidence:
                        correct_number_ef += 1

            for borough in ALL_BOROUGHS:
                pred_flagged = borough in pred_effective
                truth_flagged = borough in truth

                if pred_flagged and truth_flagged:
                    tp += 1
                    if has_real_evidence:
                        tp_ef += 1
                    else:
                        fn_ef += 1
                elif pred_flagged and not truth_flagged:
                    fp += 1
                    fp_ef += 1
                elif not pred_flagged and truth_flagged:
                    fn += 1
                    fn_ef += 1
                else:
                    tn += 1
                    tn_ef += 1

            is_failed_chunk = is_ambiguous_chunk or is_implausible_chunk
            pred_set = set(pred_effective.keys())
            truth_set = set(truth.keys())
            partial_accuracies.append(_chunk_partial_accuracy(pred_set, truth_set, is_failed_chunk))
            pred_set_ef = pred_set if has_real_evidence else set()
            partial_accuracies_ef.append(_chunk_partial_accuracy(pred_set_ef, truth_set, is_failed_chunk))

        non_ambiguous = n - n_ambiguous
        pct_with_evidence = round(evidence_weighted_sum / n, 3) if n else 0.0

        def compute_metrics(tp_, fp_, fn_, tn_):
            n_inst = tp_ + fp_ + fn_ + tn_
            accuracy = round(correct_number / n, 3) if n else 0.0
            precision = round(tp_ / (tp_ + fp_), 3) if (tp_ + fp_) > 0 else 0.0
            recall = round(tp_ / (tp_ + fn_), 3) if (tp_ + fn_) > 0 else 0.0
            f1 = round(2 * precision * recall / (precision + recall), 3) if (precision + recall) > 0 else 0.0
            return accuracy, precision, recall, f1

        accuracy, precision, recall, f1 = compute_metrics(tp, fp, fn, tn)
        # evidence-filtered accuracy uses correct_number_ef, not correct_number
        accuracy_ef = round(correct_number_ef / n, 3) if n else 0.0
        precision_ef = round(tp_ef / (tp_ef + fp_ef), 3) if (tp_ef + fp_ef) > 0 else 0.0
        recall_ef = round(tp_ef / (tp_ef + fn_ef), 3) if (tp_ef + fn_ef) > 0 else 0.0
        f1_ef = round(2 * precision_ef * recall_ef / (precision_ef + recall_ef), 3) \
            if (precision_ef + recall_ef) > 0 else 0.0

        partial_accuracy = round(sum(partial_accuracies) / n, 3) if n else 0.0
        partial_accuracy_ef = round(sum(partial_accuracies_ef) / n, 3) if n else 0.0

        rows.append({
            "mode": mode, "model": model, "n": n, "n_ambiguous": n_ambiguous,
            "non_ambiguous": non_ambiguous, "correct_number": correct_number,
            "pct_with_evidence": pct_with_evidence,
            "tp": tp, "fp": fp,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "correct_number_evidence_filtered": correct_number_ef,
            "tp_evidence_filtered": tp_ef, "fp_evidence_filtered": fp_ef,
            "accuracy_evidence_filtered": accuracy_ef,
            "precision_evidence_filtered": precision_ef,
            "recall_evidence_filtered": recall_ef,
            "f1_evidence_filtered": f1_ef,
            "partial_accuracy": partial_accuracy,
            "partial_accuracy_evidence_filtered": partial_accuracy_ef,
        })

    out = pd.DataFrame(rows).sort_values(["mode", "model"]).reset_index(drop=True)
    out = out[["mode", "model", "n", "n_ambiguous", "non_ambiguous", "correct_number",
               "pct_with_evidence", "tp", "fp", "accuracy", "precision", "recall", "f1",
               "correct_number_evidence_filtered", "tp_evidence_filtered", "fp_evidence_filtered",
               "accuracy_evidence_filtered", "precision_evidence_filtered",
               "recall_evidence_filtered", "f1_evidence_filtered",
               "partial_accuracy", "partial_accuracy_evidence_filtered"]]
    out.to_csv(OUT_CSV, index=False)

    print(out.to_string(index=False))
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    run()
