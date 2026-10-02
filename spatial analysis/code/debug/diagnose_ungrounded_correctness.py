"""
diagnose_ungrounded_correctness.py
=====================================
Traces, per row, exactly what stopped a getisord case with genuine
borough overlap from landing in ungrounded_correctness -- run this
against your actual results CSV to see whether Mechanism A
(has_real_evidence structurally near-always True) or Mechanism B
(is_implausible_chunk catching it first) is what's actually happening,
rather than guessing.

Usage:
    python diagnose_ungrounded_correctness.py
(edit RESULTS_CSV below to whichever mode's file you want to check)
"""

import pandas as pd
import error_analysis_getisord as m

RESULTS_CSV = m.RESULTS_CSV  # reuses whichever config block is active

df = pd.read_csv(RESULTS_CSV)
gt_methods = m.load_ground_truth_methods()

n_would_be_ungrounded = 0
n_caught_by_implausible = 0
n_evidence_always_true = 0

print(f"{'chunk_id':30} {'f1':>6} {'has_real_evidence':>18} {'is_implausible':>15} {'verdict'}")
print("-" * 100)

for _, r in df.iterrows():
    pred_str = r.get("predicted_boroughs")
    manifest_truth_str = r.get("ground_truth_boroughs")
    evidence_text = str(r.get("evidence_text", "")).strip().lower()

    resolved_truth_str = m.resolve_ground_truth(
        r.get("chunk_id"), r.get("statistical_method"), manifest_truth_str, gt_methods)

    pred = m.parse_borough_string(pred_str)
    truth = m.parse_borough_string(resolved_truth_str)

    is_valid_none = evidence_text == "none"
    is_empty = not isinstance(pred_str, str) or pred_str.strip() in ("", "(none)", "none") or pd.isna(pred_str)
    is_ambiguous_chunk = is_empty and not is_valid_none
    if is_ambiguous_chunk or not pred:
        continue

    raw_entries = [e.strip() for e in str(pred_str).split(";") if e.strip()]
    has_duplicates = len(raw_entries) > len(pred)
    pred_values = m.parse_borough_values(pred_str)
    borough_labels = m.parse_borough_labels(r.get("evidence_text"))
    has_implausible_value = any(not m.magnitude_plausible(v) for v in pred_values.values())
    has_inconsistent_threshold = any(
        not m.gi_threshold_consistent(v, label=borough_labels.get(b))
        for b, v in pred_values.items()
    )
    is_implausible_chunk = (
        len(pred) > m.IMPLAUSIBLE_THRESHOLD or has_duplicates
        or has_implausible_value or has_inconsistent_threshold
    )

    has_real_evidence = m._chunk_has_real_evidence(r)
    if is_valid_none and pred == truth == {}:
        has_real_evidence = True

    f1 = m._chunk_f1(pred, truth)
    if f1 <= 0:
        continue  # no overlap at all -- not a candidate for ungrounded_correctness anyway

    if is_implausible_chunk:
        n_caught_by_implausible += 1
        verdict = "CAUGHT BY is_implausible_chunk (never reaches evidence check)"
    elif has_real_evidence:
        n_evidence_always_true += 1
        verdict = "has_real_evidence=True (goes to grounded_correctness, not ungrounded)"
    else:
        n_would_be_ungrounded += 1
        verdict = ">>> THIS ONE SHOULD BE ungrounded_correctness <<<"

    print(f"{str(r.get('chunk_id'))[:30]:30} {f1:6.2f} {str(has_real_evidence):>18} "
          f"{str(is_implausible_chunk):>15} {verdict}")

print()
print("=== SUMMARY (rows with genuine borough overlap, f1 > 0) ===")
print(f"Caught by is_implausible_chunk before evidence check: {n_caught_by_implausible}")
print(f"has_real_evidence was True (correctly went to grounded_correctness): {n_evidence_always_true}")
print(f"Actually reached ungrounded_correctness: {n_would_be_ungrounded}")
