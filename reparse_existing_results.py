"""
reparse_existing_results.py
==============================
Zero-cost verification that the casual_verdict.py fix actually works --
re-runs the FIXED parser against text you ALREADY collected in results.csv
(no new API calls, no waiting). Shows exactly which rows' has_evidence
value would change under the new logic, so you can confirm the fix
without spending any time or money on a fresh LLM run.

Usage:
    python reparse_existing_results.py
Reads:  results.csv (your existing one, unchanged)
Prints: every row where has_evidence would now differ from what's stored
"""

import pandas as pd
from casual_verdict import parse_casual_with_citation

RESULTS_CSV = "results.csv"


def run():
    df = pd.read_csv(RESULTS_CSV)
    spatial = df[df["insight_type"] == "spatial_correlation"].copy()

    changed = []
    for _, r in spatial.iterrows():
        # verdict_detail holds the model's full response text (what parse_casual_with_citation needs)
        _pred, _detail, _evidence, new_has_evidence, new_cited = parse_casual_with_citation(
            r["verdict_detail"], r["spatial_granularity"])
        old_has_evidence = bool(r["has_evidence"]) if pd.notna(r["has_evidence"]) else False

        if new_has_evidence != old_has_evidence:
            changed.append({
                "chunk_id": r["chunk_id"], "model": r["model"], "mode": r["mode"],
                "old_has_evidence": old_has_evidence, "new_has_evidence": new_has_evidence,
                "verdict_detail_snippet": str(r["verdict_detail"])[:150],
            })

    if not changed:
        print("No rows changed -- either the fix doesn't affect your existing data, "
              "or something's off (check results.csv actually has spatial_correlation rows).")
        return

    print(f"{len(changed)} row(s) changed under the fixed parser:\n")
    for c in changed:
        print(f"[{c['mode']} | {c['model']} | {c['chunk_id']}]")
        print(f"  has_evidence: {c['old_has_evidence']} -> {c['new_has_evidence']}")
        print(f"  text: {c['verdict_detail_snippet']}...")
        print()

    print(f"Total spatial_correlation rows checked: {len(spatial)}")
    print(f"Rows that would flip has_evidence: {len(changed)}")


if __name__ == "__main__":
    run()
