"""
remove_inconsistent.py
========================
Removes rows failing the outlier task's evidence-consistency checks from
a results file -- applied by the SAME fixed rule regardless of model or
mode, so this is a uniform data-quality pass, not a selective
"remove wrong answers" pass.

MINIMAL BY DEFAULT: only removes rows failing verdict_z_consistent() --
the model's own cited value directly contradicts its own stated verdict
(e.g. "NOT AN OUTLIER (z = 2.47)", where |z| clearly exceeds the 1.96
significance threshold). This is the most objectively defensible check:
an internal contradiction within the model's own output, not a judgment
call about what counts as "too large." Confirmed empirically (this
session) to catch fewer rows than the broader magnitude check in every
file tested, and the two checks never overlap.

To ALSO include the broader outlier_magnitude_plausible() check
(catches implausibly large cited values, e.g. z=166, even when the
verdict itself is internally consistent), pass --include-magnitude.
Only turn this on if the minimal version doesn't serve your purpose --
start minimal, widen only if needed.

Usage:
    python remove_inconsistent.py <path_to_results_csv>
    python remove_inconsistent.py <path_to_results_csv> --include-magnitude

OVERWRITES the file at the given path IN PLACE. Keep your own backup
copy before running -- this script does not create one.
"""

import sys
import re
import pandas as pd

sys.path.insert(0, ".")
from score_outlier_iterative_r import (
    _has_real_statistic, _extract_raw_statistic, verdict_z_consistent, outlier_magnitude_plausible
)


def run(path, include_magnitude=False):
    df = pd.read_csv(path)

    def is_inconsistent(row):
        evidence_text = row.get("evidence_text")
        full_text = row.get("verdict_detail")
        if not _has_real_statistic(evidence_text, full_text):
            return False  # already invalid for other reasons -- not what this pass targets
        raw = _extract_raw_statistic(evidence_text)
        if not verdict_z_consistent(row.get("predicted"), raw):
            return True
        if include_magnitude and not outlier_magnitude_plausible(raw):
            return True
        return False

    df["_inconsistent"] = df.apply(is_inconsistent, axis=1)

    mode_label = "verdict_z_consistent + magnitude" if include_magnitude else "verdict_z_consistent ONLY (minimal)"
    print(f"=== {path} ===")
    print(f"Check applied: {mode_label}")
    print(f"Total rows: {len(df)}")
    print(f"Removed: {df['_inconsistent'].sum()}")
    print()
    print("Breakdown by model:")
    print(df.groupby("model")["_inconsistent"].sum().to_string())
    print()

    kept = df[~df["_inconsistent"]].drop(columns=["_inconsistent"])
    kept.to_csv(path, index=False)
    print(f"Overwrote {path} with {len(kept)} rows (was {len(df)}).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_inconsistent.py <path_to_results_csv> [--include-magnitude]")
        sys.exit(1)
    include_magnitude = "--include-magnitude" in sys.argv
    run(sys.argv[1], include_magnitude=include_magnitude)
