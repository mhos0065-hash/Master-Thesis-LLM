"""
fix_statistical_method_column.py
===================================
Recomputes the statistical_method column of an already-generated ReAct
results CSV, using the fixed extraction logic (blocklist for generic
labels like "test statistic"/"statistical method"/"significance
measure", plus apostrophe handling for names like "Moran's I"). Reads
evidence_text (already saved from the real run) and re-derives
statistical_method from it directly -- no API calls, no re-running any
model, nothing else in the file is touched.

Usage:
    python fix_statistical_method_column.py "ReAct/getisord_results_react.csv"
    python fix_statistical_method_column.py "ReAct/outlier_iterative_r_results_react.csv"

Overwrites the file in place. Keep a backup if you want one -- this
script does not create one itself.
"""

import re
import sys

import pandas as pd

GENERIC_NON_METHOD_LABELS = {"statistical method", "test statistic", "significance measure"}


def extract_statistical_method(evidence_text):
    """Same logic as the fixed runners: strips a leading verdict/borough
    prefix, maps known generic non-method labels to 'none', handles
    apostrophes in names like 'Moran's I'."""
    if not isinstance(evidence_text, str):
        return "none"
    match = re.search(r"([A-Za-z_\-\s\*']*)=", evidence_text)
    if not match:
        return "none"
    label = match.group(1).strip()
    # strip known verdict/borough prefixes this project's tasks use
    label = re.sub(r"^.*?(?:OUTLIER\s*-\s*(?:HIGH|LOW)|NOT AN OUTLIER)\s*\(\s*", "", label).strip()
    label = re.sub(r"^.*?(?:NOT\s+CLUSTERED|CLUSTERED)\s*\(\s*", "", label).strip()
    label = re.sub(r"^.*?\((?:HH|LL|HOT|COLD)[^,]*,\s*", "", label).strip()
    if not label or label.lower() in GENERIC_NON_METHOD_LABELS:
        return "none"
    return label


def run(path):
    df = pd.read_csv(path)
    if "evidence_text" not in df.columns or "statistical_method" not in df.columns:
        print(f"ERROR: {path} is missing evidence_text or statistical_method column -- wrong file?")
        return

    before = df["statistical_method"].copy()
    df["statistical_method"] = df["evidence_text"].apply(extract_statistical_method)
    changed = (before.astype(str) != df["statistical_method"].astype(str)).sum()

    print(f"=== {path} ===")
    print(f"Rows changed: {changed} / {len(df)}")
    if changed:
        print("\nSample of changes:")
        diff_mask = before.astype(str) != df["statistical_method"].astype(str)
        sample = df.loc[diff_mask, ["chunk_id", "model"]].copy()
        sample["before"] = before[diff_mask].values
        sample["after"] = df.loc[diff_mask, "statistical_method"].values
        print(sample.head(15).to_string(index=False))

    df.to_csv(path, index=False)
    print(f"\nOverwrote {path}.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python fix_statistical_method_column.py "<path_to_results_csv>"')
        sys.exit(1)
    run(sys.argv[1])
