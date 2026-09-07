"""
diagnose_bivariate_shift.py
==============================
Traces, PER CHUNK, exactly why the shift-invariance check disqualified
a chunk for a given model -- distinguishes a genuine value mismatch
from an extraction failure (no code found, execution crashed, or the
re-executed output's print format has no '=' sign for the fallback
extractor to find). Run this before concluding a model's 0.00
evidence rate is a real finding about the model, rather than a format
mismatch in how its code happens to print results.

Usage:
    python diagnose_bivariate_shift.py --model "deepseek/deepseek-chat-v3-0324"
"""

import argparse
import pandas as pd

from formula_invariance_check import extract_final_code, _shift_and_rerun, _is_finite, _extract_statistic
import score_bivariate as sb

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--results-csv", default=sb.RESULTS_CSV)
args = parser.parse_args()

df = pd.read_csv(args.results_csv)
df = df[df["model"] == args.model]

manifest = pd.read_csv(sb.MANIFEST_PATH).set_index("chunk_id")

print(f"{'chunk_id':35} {'result':40} {'detail'}")
print("-" * 120)

n_no_code = n_exec_failed = n_no_kv_found = n_mismatch = n_match = 0

for _, row in df.iterrows():
    chunk_id = row["chunk_id"]
    evidence_text = row.get("evidence_text", "")
    raw_stat = sb._extract_raw_statistic(evidence_text)
    if raw_stat is None or not _is_finite(raw_stat):
        print(f"{chunk_id:35} {'SKIPPED (no real original stat)':40}")
        continue

    run_dir = row["run_dir"]
    code_text = extract_final_code(run_dir)
    if not code_text:
        n_no_code += 1
        print(f"{chunk_id:35} {'NO CODE FOUND':40} extract_final_code() returned None")
        continue

    if chunk_id not in manifest.index:
        print(f"{chunk_id:35} {'CHUNK NOT IN MANIFEST':40}")
        continue
    m = manifest.loc[chunk_id]
    adjacency_path = m.get("adjacency_path")
    extra_files = {"ADJACENCY_PATH": adjacency_path} if adjacency_path and pd.notna(adjacency_path) else None

    # Re-run manually so we can inspect the RAW output before extraction,
    # not just the final True/False/None verdict.
    import formula_invariance_check as fic
    cwd_before = __import__("os").getcwd()
    import tempfile, shutil, os as _os
    tmp_root = tempfile.mkdtemp()
    raw_output = None
    try:
        rel_dir = _os.path.dirname(m["csv_path"]) or "."
        shifted_dir = _os.path.join(tmp_root, rel_dir)
        _os.makedirs(shifted_dir, exist_ok=True)
        shifted_df = pd.read_csv(m["csv_path"])
        for col in shifted_df.columns:
            if pd.api.types.is_numeric_dtype(shifted_df[col]):
                shifted_df[col] = shifted_df[col] + fic.SHIFT_CONSTANT
        shifted_csv = _os.path.join(shifted_dir, _os.path.basename(m["csv_path"]))
        shifted_df.to_csv(shifted_csv, index=False)
        if extra_files:
            for _, p in extra_files.items():
                if p and _os.path.exists(p):
                    dd = _os.path.join(tmp_root, _os.path.dirname(p) or ".")
                    _os.makedirs(dd, exist_ok=True)
                    shutil.copy(p, _os.path.join(tmp_root, p))
        run_dir_tmp = _os.path.join(tmp_root, "run")
        _os.makedirs(run_dir_tmp, exist_ok=True)
        _os.chdir(tmp_root)
        from sandbox import run_sandboxed_code
        raw_output = run_sandboxed_code(code_text, run_dir_tmp, shifted_csv, extra_vars=extra_files)
    except Exception as e:
        raw_output = None
        exec_error = str(e)
    finally:
        _os.chdir(cwd_before)
        shutil.rmtree(tmp_root, ignore_errors=True)

    if raw_output is None:
        n_exec_failed += 1
        print(f"{chunk_id:35} {'EXECUTION FAILED':40} code crashed on re-run")
        continue

    extracted = _extract_statistic(raw_output)
    if extracted is None:
        n_no_kv_found += 1
        snippet = raw_output[-150:].replace(chr(10), " | ")
        print(f"{chunk_id:35} {'NO = SIGN FOUND IN RE-RUN OUTPUT':40} tail: {snippet}")
        continue

    try:
        shifted_val = float(extracted)
        orig_val = float(raw_stat)
        denom = max(abs(orig_val), 1e-9)
        rel_diff = abs(orig_val - shifted_val) / denom
        if rel_diff <= fic.TOLERANCE:
            n_match += 1
            print(f"{chunk_id:35} {'MATCH':40} orig={orig_val:.4g} recomputed={shifted_val:.4g}")
        else:
            n_mismatch += 1
            print(f"{chunk_id:35} {'GENUINE MISMATCH':40} orig={orig_val:.4g} recomputed={shifted_val:.4g}")
    except (TypeError, ValueError):
        n_mismatch += 1
        print(f"{chunk_id:35} {'NON-FINITE RECOMPUTED VALUE':40} {extracted!r}")

print()
print("=== SUMMARY ===")
print(f"no code found:              {n_no_code}")
print(f"execution failed/crashed:   {n_exec_failed}")
print(f"no '=' sign in re-run output (format-mismatch, likely FALSE POSITIVE): {n_no_kv_found}")
print(f"genuine value mismatch:     {n_mismatch}")
print(f"matched (real evidence):    {n_match}")
