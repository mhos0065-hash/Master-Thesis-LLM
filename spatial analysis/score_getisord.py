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
    -- kept as raw, descriptive counts showing the absolute scale of
    correct/incorrect borough-level flags.
  - precision / recall / f1 are CHUNK-LEVEL MACRO-averaged: computed
    separately for EACH chunk (comparing only that chunk's predicted
    borough set against its own real set), then averaged across all n
    chunks. This is deliberately the ONLY precision/recall/F1 reported
    (not alongside a separate borough-level micro-pooled version), since
    micro-averaging was found to make F1 look misleadingly high relative
    to the strict chunk-level "correct_number" accuracy: a model that
    misses just one borough in an otherwise-correct chunk barely dents a
    pooled micro-F1 (diluted across ~700 total borough-chunk decisions),
    while that same chunk fully fails the strict exact-match accuracy
    test. Macro-averaging weights every chunk's mistake equally, which is
    much closer in spirit to what "accuracy" already measures.
    An ambiguous or implausible chunk NEVER receives perfect-match
    credit, even when ground truth happens to be empty -- its predicted
    set is empty as a CONSEQUENCE of the failure, not because the model
    deliberately confirmed "no significant boroughs" (see
    _chunk_prf1's is_failed_chunk parameter).
    Justification: the micro/macro-averaging distinction for multi-label
    classification is standard in the literature -- see Sokolova &
    Lapalme (2009), "A systematic analysis of performance measures for
    classification tasks", Information Processing & Management; Manning,
    Raghavan & Schütze (2008), "Introduction to Information Retrieval",
    Ch. 13; and Tsoumakas & Katakis (2007), "Multi-Label Classification:
    An Overview", which frames Getis-Ord-style variable-length-set
    prediction exactly as a multi-label task.
  - correct_number is CHUNK-level: a chunk only counts as correct if the
    predicted set of boroughs EXACTLY matches the real significant set.
  - n_ambiguous folds in BOTH genuinely unparseable chunks AND
    implausible ones (e.g. flagging all 33 boroughs, or a computation
    bug producing a wildly-off borough count) -- both are untrustworthy
    for the same reason, consistent with the "ambiguous counts as
    failure" principle used across every scorer this session.

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

# for multi turn
#RESULTS_CSV = "getisord_results_multi.csv"
#OUT_CSV = "Scores/multi turn/getisord_scoring_summary.csv"


#For single turn
RESULTS_CSV = "getisord_results.csv"
OUT_CSV = "Scores/single turn/getisord_scoring_summary.csv"

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
    0.0 here, same principle as correct_number and the F1 fix -- it
    never receives credit just because its forced-empty prediction
    happens to coincide with an empty ground truth.

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


def _chunk_prf1(pred_set, truth_set, is_failed_chunk=False):
    """Precision/recall/F1 for a SINGLE chunk -- comparing only this
    chunk's predicted borough set against its own real set (never pooled
    with other chunks).

    is_failed_chunk: True if this chunk was ambiguous or implausible (the
    model never gave a real, trustworthy answer). A failed chunk must
    NEVER receive perfect-match credit, even when ground truth happens to
    be empty -- pred_set is forced empty as a CONSEQUENCE of the failure,
    not because the model deliberately computed and confirmed "no
    significant boroughs". Without this check, a model that never
    answered could be scored as if it had perfectly confirmed an empty
    result, purely by coincidence (the same principle already enforced
    for accuracy and every other scorer this session -- ambiguous never
    counts as correct).

    For chunks with a real answer: if truth is empty AND prediction is
    empty, that's a genuine perfect match (1.0 all round); if truth is
    empty but prediction is NOT, that's a real mistake (0.0 all round) --
    recall is otherwise mathematically undefined (0/0) when there's
    nothing real to find."""
    if is_failed_chunk:
        return (0.0, 0.0, 0.0)

    tp = len(pred_set & truth_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)

    if not truth_set:
        return (1.0, 1.0, 1.0) if not pred_set else (0.0, 0.0, 0.0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


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

        chunk_precisions, chunk_recalls, chunk_f1s = [], [], []
        chunk_precisions_ef, chunk_recalls_ef, chunk_f1s_ef = [], [], []
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
            is_implausible_chunk = len(pred) > IMPLAUSIBLE_THRESHOLD or has_duplicates

            has_real_evidence = _chunk_has_real_evidence(r)

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

            # ---- borough-level MICRO pooling (existing) ----
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

            # ---- chunk-level MACRO precision/recall/F1 (the single,
            # reported version) -- is_failed_chunk ensures an ambiguous or
            # implausible chunk NEVER receives perfect-match credit, even
            # when ground truth happens to be empty
            is_failed_chunk = is_ambiguous_chunk or is_implausible_chunk
            pred_set = set(pred_effective.keys())
            truth_set = set(truth.keys())
            p, r_, f1_ = _chunk_prf1(pred_set, truth_set, is_failed_chunk)
            chunk_precisions.append(p)
            chunk_recalls.append(r_)
            chunk_f1s.append(f1_)

            # evidence-filtered chunk-level: an unevidenced chunk's TPs
            # don't count either
            pred_set_ef = pred_set if has_real_evidence else set()
            p_ef, r_ef, f1_ef_ = _chunk_prf1(pred_set_ef, truth_set, is_failed_chunk)
            chunk_precisions_ef.append(p_ef)
            chunk_recalls_ef.append(r_ef)
            chunk_f1s_ef.append(f1_ef_)

            # ---- Jaccard-based partial-credit accuracy (new, additive) ----
            partial_accuracies.append(_chunk_partial_accuracy(pred_set, truth_set, is_failed_chunk))
            partial_accuracies_ef.append(_chunk_partial_accuracy(pred_set_ef, truth_set, is_failed_chunk))

        non_ambiguous = n - n_ambiguous
        pct_with_evidence = round(evidence_weighted_sum / n, 3) if n else 0.0

        # accuracy_partial: RAW (non-evidence-filtered) Jaccard-based
        # partial-credit accuracy. accuracy_evidence_filtered_partial:
        # same metric, but computed only over evidence-backed
        # predictions (an unevidenced borough claim is downgraded before
        # scoring, same principle as every other evidence-filtered
        # column in this file).
        accuracy_partial = round(sum(partial_accuracies) / n, 3) if n else 0.0
        accuracy_evidence_filtered_partial = round(sum(partial_accuracies_ef) / n, 3) if n else 0.0

        precision = round(sum(chunk_precisions) / n, 3) if n else 0.0
        recall = round(sum(chunk_recalls) / n, 3) if n else 0.0
        f1 = round(sum(chunk_f1s) / n, 3) if n else 0.0
        precision_ef = round(sum(chunk_precisions_ef) / n, 3) if n else 0.0
        recall_ef = round(sum(chunk_recalls_ef) / n, 3) if n else 0.0
        f1_ef = round(sum(chunk_f1s_ef) / n, 3) if n else 0.0

        rows.append({
            "mode": mode, "model": model, "n": n, "n_ambiguous": n_ambiguous,
            "non_ambiguous": non_ambiguous, "correct_number": correct_number,
            "pct_with_evidence": pct_with_evidence,
            "tp": tp, "fp": fp,
            "accuracy_partial": accuracy_partial, "precision": precision, "recall": recall, "f1": f1,
            "correct_number_evidence_filtered": correct_number_ef,
            "tp_evidence_filtered": tp_ef, "fp_evidence_filtered": fp_ef,
            "accuracy_evidence_filtered_partial": accuracy_evidence_filtered_partial,
            "precision_evidence_filtered": precision_ef,
            "recall_evidence_filtered": recall_ef,
            "f1_evidence_filtered": f1_ef,
        })

    out = pd.DataFrame(rows).sort_values(["mode", "model"]).reset_index(drop=True)
    out = out[["mode", "model", "n", "n_ambiguous", "non_ambiguous", "correct_number",
               "pct_with_evidence", "tp", "fp", "accuracy_partial", "precision", "recall", "f1",
               "correct_number_evidence_filtered", "tp_evidence_filtered", "fp_evidence_filtered",
               "accuracy_evidence_filtered_partial", "precision_evidence_filtered",
               "recall_evidence_filtered", "f1_evidence_filtered"]]
    out.to_csv(OUT_CSV, index=False)


    print(out.to_string(index=False))
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    run()
