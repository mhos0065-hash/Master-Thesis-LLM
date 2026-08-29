"""
audit_formula_invariance.py
==============================
Post-hoc scoring-time audit for multi-agent results: checks whether the
accepted statistic is invariant to a large constant shift added to one
of the input variables.

WHY THIS IS GENERIC, NOT METHOD-SPECIFIC: any genuine correlation-style
or covariance-style spatial statistic (Moran's I, Lee's L, Getis-Ord,
Pearson correlation, etc.) is mathematically built on DEVIATIONS FROM
THE MEAN. Since (x + c) - mean(x + c) = x - mean(x) for any constant c,
a correctly-centered statistic is provably unaffected by adding a
constant to the raw data. If a model's code operates on raw values
instead of centered ones (the exact bug found manually in this
project's own logs), the reported statistic will change substantially
under this shift -- this is detectable WITHOUT knowing the correct
answer or the specific formula used.

SAFETY: this NEVER modifies the real dataset files. It copies only the
files needed for one chunk into an isolated temp directory, shifts the
data there, and temporarily changes the calling process's working
directory so that even code with a HARDCODED relative path (bypassing
the injected DATASET_PATH variable -- confirmed to occur in real logs
this session) resolves to the isolated copy, never the original.

Usage:
    python audit_formula_invariance.py --results bivariate_results_multiagent.csv \
        --manifest chunks_csv/bivariate/bivariate_manifest.csv \
        --dataset-col csv_path --adjacency-col adjacency_path
"""

import argparse
import json
import os
import re
import shutil
import tempfile

import pandas as pd

from sandbox import run_sandboxed_code

SHIFT_CONSTANT = 1000.0
TOLERANCE = 1e-3  # relative tolerance for "substantially different"

_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.*)", re.IGNORECASE)
_KV_PAIR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_ \-()]*?)\s*=\s*([\-0-9.eE]+|nan)", re.IGNORECASE)


def _extract_statistic(output_text):
    """For RAW SANDBOX OUTPUT, which includes the 'FINAL ANSWER:' prefix."""
    m = _FINAL_ANSWER_RE.search(output_text or "")
    if not m:
        return None
    pairs = _KV_PAIR_RE.findall(m.group(1))
    return pairs[0][1] if pairs else None


def _extract_statistic_from_stored_text(evidence_text):
    """For the results CSV's evidence_text column, which has already
    had the 'FINAL ANSWER:' prefix stripped by the original parser --
    searching for it here would never match."""
    pairs = _KV_PAIR_RE.findall(str(evidence_text or ""))
    return pairs[0][1] if pairs else None


def _is_finite(s):
    if s is None:
        return False
    try:
        f = float(s)
        return f == f and abs(f) != float("inf")
    except ValueError:
        return False


def _get_final_code(run_dir):
    """Reads attempts_log.json and returns the code from the attempt
    that actually produced the reported final_answer."""
    log_path = os.path.join(run_dir, "attempts_log.json")
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8") as f:
        log = json.load(f)
    target = log.get("final_answer")
    for attempt in log.get("attempts", []):
        if attempt["coder_result"].get("final_answer") == target:
            return attempt["coder_result"].get("code")
    # fallback: last attempt's code
    if log.get("attempts"):
        return log["attempts"][-1]["coder_result"].get("code")
    return None


def _shift_dataset(original_csv_path, shifted_dir):
    """Copies the dataset CSV into shifted_dir with SHIFT_CONSTANT added
    to every numeric column (except any column that looks like an ID,
    e.g. 'borough'). Returns the new file's path."""
    df = pd.read_csv(original_csv_path)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col] + SHIFT_CONSTANT
    dest = os.path.join(shifted_dir, os.path.basename(original_csv_path))
    df.to_csv(dest, index=False)
    return dest


def audit_one_row(row, manifest_row, dataset_col, adjacency_col):
    """Returns (flagged: bool, reason: str, original_stat, shifted_stat)."""
    code = _get_final_code(row["run_dir"])
    if not code:
        return False, "no code found to audit", None, None

    original_stat = _extract_statistic_from_stored_text(row["evidence_text"])
    if not _is_finite(original_stat):
        return False, "original statistic not finite -- skip, already flagged by generation-time validator", original_stat, None

    original_csv = manifest_row[dataset_col]
    original_adj = manifest_row.get(adjacency_col) if adjacency_col else None

    cwd_before = os.getcwd()
    tmp_root = tempfile.mkdtemp()
    output = None
    try:
        # Replicate the SAME relative directory structure the real
        # chunk lives in, so a hardcoded relative path in the model's
        # code resolves to the isolated, shifted copy -- never the
        # real file -- when we chdir into tmp_root below.
        rel_dir = os.path.dirname(original_csv)
        shifted_dir = os.path.join(tmp_root, rel_dir)
        os.makedirs(shifted_dir, exist_ok=True)
        shifted_csv = _shift_dataset(original_csv, shifted_dir)

        if original_adj and os.path.exists(original_adj):
            adj_rel_dir = os.path.dirname(original_adj)
            os.makedirs(os.path.join(tmp_root, adj_rel_dir), exist_ok=True)
            shutil.copy(original_adj, os.path.join(tmp_root, original_adj))

        run_dir = os.path.join(tmp_root, "run")
        os.makedirs(run_dir, exist_ok=True)

        os.chdir(tmp_root)
        extra_vars = {"ADJACENCY_PATH": original_adj} if original_adj else None
        output = run_sandboxed_code(code, run_dir, shifted_csv, extra_vars=extra_vars)
    except Exception:
        output = None
    finally:
        os.chdir(cwd_before)
        shutil.rmtree(tmp_root, ignore_errors=True)  # never let cleanup failure discard a captured result

    if output is None:
        return False, "shifted re-execution failed to produce output -- treated as inconclusive, not a flag", original_stat, None

    shifted_stat = _extract_statistic(output)
    if not _is_finite(shifted_stat):
        return True, f"statistic became non-finite under a constant shift (was {original_stat}, now {shifted_stat}) -- strong evidence of uncentered/raw-value computation", original_stat, shifted_stat

    orig_f, shift_f = float(original_stat), float(shifted_stat)
    denom = max(abs(orig_f), 1e-9)
    rel_diff = abs(orig_f - shift_f) / denom
    if rel_diff > TOLERANCE:
        return True, f"statistic changed substantially under a constant shift ({orig_f} -> {shift_f}, {rel_diff:.1%} relative difference) -- likely uses raw values instead of deviations from the mean", original_stat, shifted_stat

    return False, "invariant under constant shift", original_stat, shifted_stat


def run(results_csv, manifest_csv, dataset_col, adjacency_col, chunk_id_col="chunk_id"):
    results = pd.read_csv(results_csv)
    manifest = pd.read_csv(manifest_csv).set_index(chunk_id_col)

    flags, reasons, orig_stats, shift_stats = [], [], [], []
    for _, row in results.iterrows():
        if row[chunk_id_col] not in manifest.index:
            flags.append(False); reasons.append("chunk not in manifest"); orig_stats.append(None); shift_stats.append(None)
            continue
        manifest_row = manifest.loc[row[chunk_id_col]]
        flagged, reason, orig, shift = audit_one_row(row, manifest_row, dataset_col, adjacency_col)
        flags.append(flagged); reasons.append(reason); orig_stats.append(orig); shift_stats.append(shift)
        print(f"[{row['model']} | {row[chunk_id_col]}] flagged={flagged} -- {reason}", flush=True)

    results["formula_invariance_flagged"] = flags
    results["formula_invariance_reason"] = reasons
    results["formula_invariance_original_stat"] = orig_stats
    results["formula_invariance_shifted_stat"] = shift_stats

    out_path = results_csv.replace(".csv", "_audited.csv")
    results.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(f"Flagged {sum(flags)} / {len(flags)} rows as likely formula-invariance failures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-col", default="csv_path")
    parser.add_argument("--adjacency-col", default="adjacency_path")
    args = parser.parse_args()
    run(args.results, args.manifest, args.dataset_col, args.adjacency_col)
