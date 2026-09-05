"""
diagnose_getisord_gates.py
=============================
Traces, PER CHUNK, exactly which of the three possible gates
disqualified it for a given model -- run this against your real files
to see whether low pct_with_evidence/accuracy for a model comes from
(a) genuinely implausible values (a real model limitation), (b) the
shift-invariance re-check specifically, or (c) something else. Reuses
the REAL functions from score_getisord.py and statistic_plausibility.py
-- no stubbing, so this reflects exactly what the real scorer does.

Usage:
    python diagnose_getisord_gates.py --model "meta-llama/llama-3.1-8b-instruct"
"""

import argparse
import pandas as pd

import score_getisord as sg

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--results-csv", default=sg.RESULTS_CSV)
args = parser.parse_args()

df = pd.read_csv(args.results_csv)
df = df[df["model"] == args.model]

gt_methods = sg.load_ground_truth_methods()
manifest = sg._load_manifest_index()

print(f"{'chunk_id':30} {'gate_that_fired':30} {'detail'}")
print("-" * 100)

for _, r in df.iterrows():
    chunk_id = r.get("chunk_id")
    pred_str = r.get("predicted_boroughs")
    manifest_truth_str = r.get("ground_truth_boroughs")
    evidence_text = str(r.get("evidence_text", "")).strip().lower()

    resolved_truth_str, gt_source, gt_method_used = sg.resolve_ground_truth(
        chunk_id, r.get("statistical_method"), manifest_truth_str, gt_methods)

    pred = sg.parse_borough_string(pred_str)
    truth = sg.parse_borough_string(resolved_truth_str)

    is_valid_none = evidence_text == "none"
    is_empty = not isinstance(pred_str, str) or pred_str.strip() in ("", "(none)", "none") or pd.isna(pred_str)
    is_ambiguous_chunk = is_empty and not is_valid_none

    raw_entries = [e.strip() for e in str(pred_str).split(";") if e.strip()] if not is_empty else []
    has_duplicates = len(raw_entries) > len(pred)

    pred_values = sg.parse_borough_values(pred_str)
    borough_labels = sg.parse_borough_labels(r.get("evidence_text"))
    has_implausible_value = any(not sg.magnitude_plausible(v) for v in pred_values.values())
    has_inconsistent_threshold = any(
        not sg.gi_threshold_consistent(v, label=borough_labels.get(b))
        for b, v in pred_values.items()
    )
    is_implausible_chunk = (
        len(pred) > sg.IMPLAUSIBLE_THRESHOLD or has_duplicates
        or has_implausible_value or has_inconsistent_threshold
    )

    has_real_evidence = sg._chunk_has_real_evidence(r)
    if is_valid_none and pred == truth == {}:
        has_real_evidence = True
    pre_shift_evidence = has_real_evidence

    shift_result = "not checked"
    if has_real_evidence and pred_values and chunk_id in manifest.index:
        m = manifest.loc[chunk_id]
        shifted_stat = sg.compute_shifted_statistic(r.get("run_dir"), m["csv_path"], m.get("adjacency_path"))
        if shifted_stat is None:
            shift_result = "inconclusive (no code found / exec failed)"
        else:
            try:
                shifted_val = float(shifted_stat)
                tol = sg.METHOD_TOLERANCE.get(gt_method_used, sg.DEFAULT_TOLERANCE)
                is_match = any(abs(shifted_val - v) <= tol for v in pred_values.values())
                shift_result = f"MATCH (recomputed={shifted_val:.4g})" if is_match else \
                    f"MISMATCH (recomputed={shifted_val:.4g}, originals={list(pred_values.values())})"
                if not is_match:
                    has_real_evidence = False
            except (TypeError, ValueError):
                shift_result = "non-finite recomputed value"
                has_real_evidence = False

    # Determine which gate is the ROOT CAUSE of exclusion, in priority order
    if is_ambiguous_chunk:
        gate, detail = "AMBIGUOUS (empty/no answer)", f"pred_str={pred_str!r}"
    elif is_implausible_chunk:
        reasons = []
        if len(pred) > sg.IMPLAUSIBLE_THRESHOLD:
            reasons.append(f"too many boroughs ({len(pred)})")
        if has_duplicates:
            reasons.append("duplicate entries")
        if has_implausible_value:
            reasons.append(f"implausible magnitude {list(pred_values.values())}")
        if has_inconsistent_threshold:
            reasons.append(f"value/label mismatch (e.g. near-zero labeled HOT/COLD): {list(pred_values.values())}")
        gate, detail = "IMPLAUSIBLE_CHUNK", "; ".join(reasons)
    elif not pre_shift_evidence:
        gate, detail = "NO REAL STATISTIC CITED", f"evidence_text={r.get('evidence_text', '')[:80]!r}"
    elif not has_real_evidence:
        gate, detail = "SHIFT-INVARIANCE MISMATCH", shift_result
    else:
        exact = set(pred.keys()) == set(truth.keys())
        gate, detail = ("EXACT MATCH" if exact else "PARTIAL/WRONG (evidenced OK)"), \
            f"pred={set(pred.keys())} truth={set(truth.keys())}"

    print(f"{str(chunk_id)[:30]:30} {gate:30} {detail}")
