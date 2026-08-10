"""
run_missing_chunks.py
========================
Finds which chunks are MISSING from results.csv for a given model+mode
(i.e. never successfully completed), and runs ONLY those -- avoids
duplicating chunks that already have a real result, useful for resuming a
sweep that was interrupted partway through (e.g. by a crash that's since
been fixed).

Usage:
    python run_missing_chunks.py --model deepseek/deepseek-chat-v3-0324 --mode single
    python run_missing_chunks.py --model deepseek/deepseek-chat-v3-0324 --mode multi
"""

import argparse
import os
import pandas as pd

import single_agent
import multi_agent
from manifest_io import load_manifest

MANIFEST_PATH = "chunks_csv/manifest.csv"
RESULTS_CSV = "results.csv"


def get_completed_chunks(model, mode):
    """Returns the set of chunk_ids that already have a row for this
    model+mode in results.csv (regardless of whether predicted succeeded
    or was None -- a row existing means the run finished without crashing,
    which is what we care about for resuming)."""
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    subset = df[(df["model"] == model) & (df["mode"] == mode)]
    return set(subset["chunk_id"])


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    args = parser.parse_args()

    manifest = load_manifest(MANIFEST_PATH)
    all_chunks = set(manifest.keys())
    completed = get_completed_chunks(args.model, args.mode)
    missing = sorted(all_chunks - completed)

    print(f"Model: {args.model} | Mode: {args.mode}")
    print(f"Total chunks: {len(all_chunks)} | Already completed: {len(completed)} | Missing: {len(missing)}")

    if not missing:
        print("Nothing missing -- this model+mode is already fully complete.")
        return

    print(f"\nMissing chunks:\n" + "\n".join(f"  {c}" for c in missing))
    print()

    run_fn = single_agent.run_one if args.mode == "single" else multi_agent.run_one
    failed = 0
    for cid in missing:
        try:
            run_fn(cid, manifest[cid], args.model)
        except Exception as e:
            failed += 1
            print(f"[{args.mode} | {args.model} | {cid}] CRASHED: {type(e).__name__}: {e}")

    print(f"\nDone. {failed} of {len(missing)} missing chunk(s) still failed.")


if __name__ == "__main__":
    run()
