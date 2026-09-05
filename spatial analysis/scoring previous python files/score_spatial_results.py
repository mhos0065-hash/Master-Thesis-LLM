"""
score_spatial_results.py
==========================
Scores results.csv (single/multi-agent, Global + Local Moran's I) with
the SAME column structure used across every other task's scorer.

GLOBAL rows: unchanged from the previous version -- verdict correctness
resolved per row against the method-matched ground truth (from
ground_truth_all_methods_spatial.csv), boolean CLUSTERED/NOT_CLUSTERED,
2-class macro precision/recall/F1.

LOCAL rows: CHANGED. The task's own prompt asks for SPECIFIC named
boroughs ("HOT SPOT: a borough ... COLD SPOT: a borough ..."), not
just a yes/no existence check -- so scoring only the derived TRUE/FALSE
boolean and ignoring which boroughs were actually named under-tests
what was asked. A model could name completely wrong boroughs and still
score as fully correct under a pure existence check.

Local correctness is now per-chunk PARTIAL CREDIT: the model's cited
borough names (parsed from evidence_text) are compared as a SET against
the manifest's `significant_cluster_boroughs` (the same ground truth
this task always had, just not previously checked at this level of
detail), and scored via per-chunk F1 -- credit for each correctly
named borough, a penalty for each wrong one (both folded into the
standard precision/recall/F1 formula), never negative (F1 is naturally
bounded in [0, 1], so no explicit floor is needed the way a raw
credit-minus-penalty score would require one). This is the SAME
approach used for getisord's per-chunk F1, chosen there for the same
reason: it penalizes false positives and false negatives symmetrically
without being diluted by an unrelated large negative pool, unlike
Informedness/Youden's J.

NO NEW COLUMNS: `correct` becomes a FLOAT in [0, 1] for local rows
(instead of a strict boolean) and stays exactly as before (0/1) for
global rows -- `accuracy` (= mean of `correct` across all chunks) and
the pooled `tp`/`fp`/`precision`/`recall`/`f1` columns are computed
from the SAME formulas as before, they just now blend exact-match
credit (global) with partial credit (local) under the hood. This is
implemented by treating "LOCAL_BOROUGH_MATCH" as a THIRD pseudo-class
alongside the existing CLUSTERED/NOT_CLUSTERED classes in the macro-
average -- its tp/fp/fn are borough-level counts pooled across every
local chunk in the group, averaged in on equal footing with the two
existing chunk-level classes. See the scoring-logic table in the chat
for the plain-English version of this.

Evidence check: unchanged -- a chunk only counts as evidenced if at
least one REAL, finite, non-placeholder number is cited.

Usage:
    python score_spatial_results.py
Reads:  <mode>_results_with_method.csv (must include `statistical_method`
        -- see attach_statistical_method_spatial.py),
        ground_truth_all_methods_spatial.csv, manifest.csv
Writes: scoring_summary.csv
"""

import json
import re
import pandas as pd

from statistic_plausibility import magnitude_plausible, bounded_probability, verdict_p_consistent

GROUND_TRUTH_METHODS_PATH = "ground truth/ground_truth_all_methods_spatial.csv"
MANIFEST_PATH = "chunks_csv/manifest.csv"

# Multi agent
#RESULTS_CSV = "results/multi agent/spatial_correlation_results_multiagent_with_method.csv"
#OUT_CSV = "Scores/multi agent/scoring_spatial_summary.csv"

# Multi turn
#RESULTS_CSV = "results/single agent/results_spatial_multi_with_method.csv"
#OUT_CSV = "Scores/multi turn/scoring_spatial_summary.csv"

# Single turn
RESULTS_CSV = "spatial_results_with_method.csv"
OUT_CSV = "Scores/single turn/scoring_spatial_summary.csv"

METHOD_NAME_MAP_GLOBAL = {
    "Global Moran's I": "Global Moran's I",
    "Geary's C": "Geary's C",
    "Join Count Statistic": "Join Count Statistic",
    "Neighbor-average correlation (non-standard)": "Neighbor-average correlation (non-standard)",
}
METHOD_NAME_MAP_LOCAL = {
    "Local Moran's I / LISA": "Local Moran's I / LISA",
    "Getis-Ord Gi / Gi*": "Getis-Ord Gi / Gi*",
}

VALUE_TOLERANCE = 0.05  # flat absolute, GLOBAL rows only


def _has_real_statistic(evidence_text):
    if not isinstance(evidence_text, str):
        return False
    match = re.search(r"=\s*([^,)]+)[,)]", evidence_text)
    if not match:
        return False
    val = match.group(1).strip()
    try:
        f = float(val)
        return f == f and abs(f) != float("inf")
    except ValueError:
        return False


def parse_bool(v):
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in ("true", "found", "1", "yes", "clustered", "present"):
        return True
    if s in ("false", "not found", "0", "no", "not clustered", "not present", "random"):
        return False
    return None


def _extract_raw_statistic(evidence_text):
    if not isinstance(evidence_text, str):
        return None
    match = re.search(r"=\s*([^,)]+)[,)]", evidence_text)
    return match.group(1).strip() if match else None


def _extract_p_value(evidence_text):
    if not isinstance(evidence_text, str):
        return None
    matches = re.findall(r"=\s*([^,)]+)[,)]", evidence_text)
    return matches[1].strip() if len(matches) >= 2 else None


def _extract_local_values(evidence_text):
    if not isinstance(evidence_text, str):
        return []
    vals = []
    for raw in re.findall(r"=\s*([^,)]+)[,)]", evidence_text):
        try:
            f = float(raw.strip())
            if f == f and abs(f) != float("inf"):
                vals.append(f)
        except ValueError:
            continue
    return vals


def _normalize_borough_name(name):
    """Normalizes a borough name for comparison -- NOT an exact-string
    match, per explicit instruction. Lowercases, strips whitespace,
    collapses multiple spaces, and normalizes common variant spellings
    ('&' vs 'and', stray punctuation) so that e.g. 'Kensington & Chelsea'
    and 'Kensington and Chelsea' -- or trailing/leading whitespace, or
    a stray period -- are treated as the SAME borough rather than
    silently counted as a miss and a false positive."""
    n = name.strip().lower()
    n = n.replace("&", "and")
    n = re.sub(r"[.\u2019']", "", n)  # stray periods / apostrophes
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _extract_local_borough_names(evidence_text):
    """LOCAL rows list boroughs as 'BoroughName (HH, I=1.2), Other (LL,
    I=-0.3)' -- extracts just the NAMES (presence-only, no HH/LL
    direction check, matching the presence-only convention already
    used for getisord). Returns a set of NORMALIZED names."""
    if not isinstance(evidence_text, str):
        return set()
    names = set()
    for entry in re.findall(r"([A-Za-z][A-Za-z .'\-&]*?)\s*\(", evidence_text):
        name = entry.strip()
        if name:
            names.add(_normalize_borough_name(name))
    return names


def _to_float(raw):
    if raw is None:
        return None
    try:
        f = float(str(raw).strip())
        return f if (f == f and abs(f) != float("inf")) else None
    except ValueError:
        return None


def load_ground_truth_methods():
    gt = pd.read_csv(GROUND_TRUTH_METHODS_PATH)
    table = {}
    for _, r in gt.iterrows():
        table[(r["chunk_id"], r["method"])] = {
            "gt_stat": r["gt_stat"] if pd.notna(r["gt_stat"]) else None,
            "gt_clustered": bool(r["gt_clustered"]),
        }
    return table


def load_local_borough_ground_truth():
    """chunk_id -> set of borough names in the manifest's
    significant_cluster_boroughs (the SAME ground truth this task
    always had -- ground_truth_metric_json -- just not previously
    checked at borough-name granularity)."""
    manifest = pd.read_csv(MANIFEST_PATH)
    table = {}
    for _, r in manifest.iterrows():
        if str(r.get("spatial_granularity", "")).strip().lower() != "local":
            continue
        raw = r.get("ground_truth_metric_json")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else {}
            boroughs = parsed.get("significant_cluster_boroughs", {})
            table[r["chunk_id"]] = {_normalize_borough_name(b) for b in boroughs.keys()}
        except (ValueError, TypeError):
            table[r["chunk_id"]] = set()
    return table


def resolve_ground_truth(chunk_id, statistical_method, granularity, manifest_clustered, gt_methods):
    method_map = METHOD_NAME_MAP_GLOBAL if granularity == "global" else METHOD_NAME_MAP_LOCAL
    fallback_method = "Global Moran's I" if granularity == "global" else "Local Moran's I / LISA"

    normalized_method = method_map.get(statistical_method)
    if normalized_method is not None:
        key = (chunk_id, normalized_method)
        if key in gt_methods:
            gt = gt_methods[key]
            return gt["gt_clustered"], gt["gt_stat"], "method_specific", normalized_method

    return bool(manifest_clustered), None, "manifest_fallback", fallback_method


def value_within_tolerance(model_val, gt_val):
    m = _to_float(model_val)
    g = _to_float(gt_val)
    if m is None or g is None:
        return None
    return abs(m - g) <= VALUE_TOLERANCE


def _local_chunk_f1(pred_set, true_set):
    """Per-chunk F1 between the model's cited borough names and the
    manifest's true significant_cluster_boroughs -- credit for each
    correct name, penalty for each wrong one, naturally bounded [0, 1]
    (F1 can't go negative, so no separate floor is needed). Correctly-
    empty (both sets empty, e.g. a genuine 'no significant boroughs'
    chunk matched by a genuine 'none' answer) scores 1.0."""
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, tp, fp, fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, tp, fp, fn


def load():
    df = pd.read_csv(RESULTS_CSV)

    if "statistical_method" not in df.columns:
        raise ValueError(
            f"'{RESULTS_CSV}' has no `statistical_method` column. "
            "Run attach_statistical_method_spatial.py first and point "
            "RESULTS_CSV at its '_with_method' output."
        )

    gt_methods = load_ground_truth_methods()
    local_borough_gt = load_local_borough_ground_truth()

    resolved = df.apply(
        lambda r: resolve_ground_truth(
            r["chunk_id"], r.get("statistical_method"), r["spatial_granularity"],
            r["ground_truth"], gt_methods),
        axis=1)
    df["gt_clustered_used"] = resolved.apply(lambda t: t[0])
    df["gt_stat_used"] = resolved.apply(lambda t: t[1])
    df["ground_truth_source"] = resolved.apply(lambda t: t[2])
    df["gt_method_used"] = resolved.apply(lambda t: t[3])

    df["predicted_bool"] = df["predicted"].apply(parse_bool)
    df["ambiguous"] = df["predicted_bool"].isna()
    df["has_real_statistic_bool"] = df["evidence_text"].apply(_has_real_statistic)
    df["is_local"] = df["spatial_granularity"].astype(str).str.strip().str.lower() == "local"

    def _plausibility_check(row):
        if not row["has_real_statistic_bool"]:
            return True
        if row["is_local"]:
            values = _extract_local_values(row["evidence_text"])
            return all(magnitude_plausible(v) for v in values)
        else:
            raw_stat = _extract_raw_statistic(row["evidence_text"])
            raw_p = _extract_p_value(row["evidence_text"])
            if not magnitude_plausible(raw_stat):
                return False
            if not bounded_probability(raw_p):
                return False
            if not pd.isna(row["predicted_bool"]):
                if not verdict_p_consistent(bool(row["predicted_bool"]), raw_p):
                    return False
            return True

    df["statistic_plausible"] = df.apply(_plausibility_check, axis=1)
    df["has_evidence_bool"] = df["has_real_statistic_bool"] & df["statistic_plausible"]

    # --- correctness: boolean for GLOBAL, partial-credit F1 for LOCAL ---
    def _row_correctness(row):
        if row["is_local"]:
            if row["ambiguous"]:
                return 0.0, 0, 0, 0
            pred_names = _extract_local_borough_names(row["evidence_text"])
            true_names = local_borough_gt.get(row["chunk_id"], set())
            f1, tp_c, fp_c, fn_c = _local_chunk_f1(pred_names, true_names)
            return f1, tp_c, fp_c, fn_c
        else:
            if row["ambiguous"]:
                return 0.0, 0, 0, 0
            is_correct = row["predicted_bool"] == row["gt_clustered_used"]
            return (1.0 if is_correct else 0.0), None, None, None

    correctness = df.apply(_row_correctness, axis=1)
    df["correct"] = correctness.apply(lambda t: t[0])
    df["local_tp"] = correctness.apply(lambda t: t[1])
    df["local_fp"] = correctness.apply(lambda t: t[2])
    df["local_fn"] = correctness.apply(lambda t: t[3])

    df["correct_evidence_filtered"] = df["correct"] * df["has_evidence_bool"].astype(float)

    def _value_check(row):
        if row["is_local"] or row["correct"] < 1.0 or not row["has_evidence_bool"]:
            return None
        if row["gt_stat_used"] is None:
            return None
        raw_stat = _extract_raw_statistic(row["evidence_text"])
        return value_within_tolerance(raw_stat, row["gt_stat_used"])

    df["value_verified"] = df.apply(_value_check, axis=1)
    df["fully_verified"] = (
        (df["correct"] >= 1.0) & df["has_evidence_bool"]
        & ((df["value_verified"] == True) | (df["is_local"] & (df["correct"] >= 1.0)))  # noqa: E712
    )

    return df


def compute_metrics(g, use_evidence_filter):
    """3-class macro average: CLUSTERED, NOT_CLUSTERED (both from GLOBAL
    rows, pooled at chunk level, exactly as before), and
    LOCAL_BOROUGH_MATCH (from LOCAL rows, pooled at BOROUGH level across
    every local chunk in the group). All three are averaged with equal
    weight if they occur at all in this group -- same "only classes
    that actually occur count toward the macro average" principle used
    everywhere else in this pipeline. tp/fp returned are the SUM across
    all three (chunk-level global counts + borough-level local counts)
    -- an aggregate diagnostic, not a single pure statistic, since it
    mixes two different levels of granularity; see scoring-logic table."""
    is_local = g["is_local"]
    g_global = g[~is_local]
    g_local = g[is_local]

    use_col = "correct_evidence_filtered" if use_evidence_filter else "correct"

    tp_total = fp_total = 0
    precisions, recalls, f1s = [], [], []

    # --- GLOBAL: CLUSTERED / NOT_CLUSTERED classes, chunk-level ---
    if len(g_global) > 0:
        y_true = g_global["gt_clustered_used"].astype(bool)
        has_evidence = g_global["has_evidence_bool"].astype(bool)
        if use_evidence_filter:
            keep_as_is = g_global["predicted_bool"].isna() | (g_global["predicted_bool"] == False) | has_evidence  # noqa: E712
            y_pred = g_global["predicted_bool"].where(keep_as_is, None)
        else:
            y_pred = g_global["predicted_bool"]

        for target_class in (True, False):
            n_real = int((y_true == target_class).sum())
            if n_real == 0:
                continue
            tp_c = int(((y_pred == target_class) & (y_true == target_class)).sum())
            fp_c = int(((y_pred == target_class) & (y_true != target_class)).sum())
            fn_c = int(((y_pred != target_class) & (y_true == target_class)).sum())
            tp_total += tp_c
            fp_total += fp_c
            p = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            r = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            f1_c = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1_c)

    # --- LOCAL: LOCAL_BOROUGH_MATCH class, borough-level, pooled ---
    if len(g_local) > 0:
        if use_evidence_filter:
            counted = g_local["has_evidence_bool"]
        else:
            counted = pd.Series(True, index=g_local.index)
        tp_l = int(g_local.loc[counted, "local_tp"].fillna(0).sum())
        fp_l = int(g_local.loc[counted, "local_fp"].fillna(0).sum())
        fn_l = int(g_local.loc[counted, "local_fn"].fillna(0).sum())
        # rows filtered out by evidence still contribute their true
        # boroughs as misses (fn), same "unevidenced = treated as a
        # miss" principle used for global's evidence filtering
        if use_evidence_filter:
            fn_l += int(g_local.loc[~counted, "local_fn"].fillna(0).sum()) \
                + int(g_local.loc[~counted, "local_tp"].fillna(0).sum())

        tp_total += tp_l
        fp_total += fp_l
        if tp_l + fp_l + fn_l > 0:
            p = tp_l / (tp_l + fp_l) if (tp_l + fp_l) > 0 else 0.0
            r = tp_l / (tp_l + fn_l) if (tp_l + fn_l) > 0 else 0.0
            f1_l = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1_l)

    precision = sum(precisions) / len(precisions) if precisions else 0.0
    recall = sum(recalls) / len(recalls) if recalls else 0.0
    f1 = sum(f1s) / len(f1s) if f1s else 0.0

    n_total = len(g)
    accuracy = g[use_col].sum() / n_total if n_total > 0 else 0.0
    return round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3), tp_total, fp_total


def run():
    df = load()
    rows = []

    for (mode, model), g in df.groupby(["mode", "model"]):
        n = len(g)
        n_ambiguous = int(g["ambiguous"].sum())
        non_ambiguous = n - n_ambiguous
        correct_number = round(g["correct"].sum(), 3)
        correct_number_ef = round(g["correct_evidence_filtered"].sum(), 3)
        pct_with_evidence = round(g["has_evidence_bool"].mean(), 3)

        if n == 0:
            continue

        accuracy, precision, recall, f1, tp, fp = compute_metrics(g, use_evidence_filter=False)
        acc_ef, prec_ef, rec_ef, f1_ef, tp_ef, fp_ef = compute_metrics(g, use_evidence_filter=True)

        rows.append({
            "mode": mode, "model": model, "n": n, "n_ambiguous": n_ambiguous,
            "non_ambiguous": non_ambiguous, "correct_number": correct_number,
            "pct_with_evidence": pct_with_evidence,
            "tp": tp, "fp": fp,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "correct_number_evidence_filtered": correct_number_ef,
            "tp_evidence_filtered": tp_ef, "fp_evidence_filtered": fp_ef,
            "accuracy_evidence_filtered": acc_ef,
            "precision_evidence_filtered": prec_ef,
            "recall_evidence_filtered": rec_ef,
            "f1_evidence_filtered": f1_ef,
        })

    out = pd.DataFrame(rows).sort_values(["mode", "model"]).reset_index(drop=True)
    out = out[["mode", "model", "n", "n_ambiguous", "non_ambiguous", "correct_number",
               "pct_with_evidence", "tp", "fp", "accuracy", "precision", "recall", "f1",
               "correct_number_evidence_filtered", "tp_evidence_filtered", "fp_evidence_filtered",
               "accuracy_evidence_filtered", "precision_evidence_filtered",
               "recall_evidence_filtered", "f1_evidence_filtered"]]
    out.to_csv(OUT_CSV, index=False)

    print(out.to_string(index=False))
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    run()
