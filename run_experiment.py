"""
run_experiment.py
===================
Convenience driver: runs BOTH single_agent and multi_agent pipelines,
across a list of models, over every chunk in the manifest -- i.e. the
fully-crossed design from Section 4.6 of your proposal (model family x
pipeline architecture x insight type), scaled down to your free-tier model
list for the first pass.

Usage:
    python3 run_experiment.py
    python3 run_experiment.py --models llama-3.3-70b-versatile qwen/qwen3-32b
    python3 run_experiment.py --chunks cyclehire_outlier_sample leggi_attribution_cityoflondon

Results all accumulate in results.csv (shared with single_agent.py /
multi_agent.py) -- run score_results.py afterwards to get precision/recall/F1.

RESILIENT TO CRASHES: each (chunk, model, mode) run is wrapped in try/except.
If one run throws an unexpected error (e.g. an encoding issue, a transient
API failure, a model returning something unparseable), it's logged to
failed_runs.log and the sweep CONTINUES to the next run, rather than losing
all remaining progress -- a single bad run used to kill the entire script.
"""

import argparse
import traceback
from datetime import datetime, timezone

import single_agent
import multi_agent
from manifest_io import load_manifest

DEFAULT_MODELS = [
    "llama-3.3-70b-versatile",   # Groq free tier, generalist, high-capability
    "qwen/qwen3-32b",            # Groq free tier, reasoning-augmented
]

MANIFEST_PATH = "chunks_csv/manifest.csv"
FAILED_RUNS_LOG = "failed_runs.log"


def safe_run(fn, cid, meta, model, mode_label):
    """Runs fn(cid, meta, model), catching and logging any exception instead
    of letting it propagate and kill the whole sweep."""
    try:
        fn(cid, meta, model)
        return True
    except Exception as e:
        ts = datetime.now(timezone.utc).isoformat()
        with open(FAILED_RUNS_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n[{ts}] FAILED: mode={mode_label} model={model} chunk={cid}\n")
            f.write(traceback.format_exc())
            f.write("\n" + "-" * 70 + "\n")
        print(f"[{mode_label} | {model} | {cid}] CRASHED -- logged to {FAILED_RUNS_LOG}, continuing...")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--chunks", nargs="+", default=None, help="subset of chunk_ids (default: all)")
    args = parser.parse_args()

    manifest = load_manifest(MANIFEST_PATH)
    chunk_ids = args.chunks or list(manifest.keys())

    total = len(args.models) * len(chunk_ids) * 2
    done, failed = 0, 0
    for model in args.models:
        for cid in chunk_ids:
            meta = manifest[cid]
            if not safe_run(single_agent.run_one, cid, meta, model, "single"):
                failed += 1
            done += 1
            print(f"progress: {done}/{total} ({failed} failed so far)")

            if not safe_run(multi_agent.run_one, cid, meta, model, "multi"):
                failed += 1
            done += 1
            print(f"progress: {done}/{total} ({failed} failed so far)")

    print(f"\nAll runs complete. {failed} run(s) failed (see {FAILED_RUNS_LOG} if >0). "
          f"Run: python3 score_results.py")
