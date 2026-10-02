"""
Diagnostic check for LLM experiment results CSVs — run this BEFORE scoring
to catch data quality problems that would otherwise silently contaminate
your accuracy metrics.

Checks for three kinds of bad rows:
  1. Total failures    - predicted is empty/NaN (model gave no verdict at all)
  2. Broken templates   - model left a placeholder like "<value>" unfilled,
                          or asked for more information instead of answering
  3. Fabricated evidence - the model's cited slope number does NOT match the
                          real ground_truth_slope, even when it got the
                          category label right. This is the dangerous one:
                          it looks like a valid, evidenced result unless you
                          check the actual numbers.

USAGE:
    python check_results.py distance_decay_results.csv
    python check_results.py distance_decay_results.csv --tolerance 0.3
    python check_results.py distance_decay_results.csv --save-flagged
"""
import argparse
import re
import sys
import pandas as pd


def extract_number(text, field_name):
    """Pull a number like 'slope = -3.21' out of the evidence text."""
    if pd.isna(text):
        return None
    match = re.search(rf"{field_name}\s*=\s*(-?[\d.]+)", str(text))
    return float(match.group(1)) if match else None


def check_results(csv_path, tolerance=0.5):
    df = pd.read_csv(csv_path)

    required_cols = {"chunk_id", "model", "predicted", "ground_truth_direction",
                      "ground_truth_slope", "evidence_text"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"ERROR: CSV is missing expected columns: {missing_cols}")
        sys.exit(1)

    df["evidence_slope"] = df["evidence_text"].apply(lambda t: extract_number(t, "slope"))
    df["slope_diff"] = (df["evidence_slope"] - df["ground_truth_slope"]).abs()

    df["issue_total_failure"] = df["predicted"].isna()
    df["issue_broken_template"] = df["evidence_text"].str.contains(
        "<value>|please provide", case=False, na=False)
    df["issue_fabricated_evidence"] = (
        df["evidence_slope"].notna() & (df["slope_diff"] > tolerance)
    )
    df["issue_any"] = (
        df["issue_total_failure"] | df["issue_broken_template"] | df["issue_fabricated_evidence"]
    )
    df["label_correct"] = df["predicted"] == df["ground_truth_direction"]

    # ---- Overall summary ----
    total = len(df)
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC REPORT: {csv_path}")
    print(f"{'='*70}")
    print(f"Total rows: {total}")
    print(f"  Total failures (no verdict):     {df['issue_total_failure'].sum()}")
    print(f"  Broken/unfilled templates:       {df['issue_broken_template'].sum()}")
    print(f"  Fabricated evidence (>{tolerance} off):  {df['issue_fabricated_evidence'].sum()}")
    print(f"  ANY problem:                      {df['issue_any'].sum()} "
          f"({df['issue_any'].sum() / total * 100:.1f}%)")
    print(f"  Clean rows remaining:             {(~df['issue_any']).sum()} "
          f"({(~df['issue_any']).sum() / total * 100:.1f}%)")

    # ---- Per-model breakdown ----
    print(f"\n{'-'*70}")
    print("PER-MODEL BREAKDOWN")
    print(f"{'-'*70}")
    summary = df.groupby("model").agg(
        total=("chunk_id", "count"),
        total_failures=("issue_total_failure", "sum"),
        broken_template=("issue_broken_template", "sum"),
        fabricated_evidence=("issue_fabricated_evidence", "sum"),
        any_problem=("issue_any", "sum"),
        label_correct=("label_correct", "sum"),
    ).reset_index()
    summary["naive_accuracy_pct"] = (summary["label_correct"] / summary["total"] * 100).round(1)
    summary["problem_pct"] = (summary["any_problem"] / summary["total"] * 100).round(1)
    summary["clean_accuracy_candidates"] = summary["total"] - summary["any_problem"]
    summary = summary.sort_values("problem_pct")
    print(summary.to_string(index=False))

    # ---- Worst offenders: fabricated evidence with suspiciously repeated values ----
    fabricated = df[df["issue_fabricated_evidence"]]
    if len(fabricated) > 0:
        print(f"\n{'-'*70}")
        print("SUSPICIOUS REPEATED PLACEHOLDER VALUES (same slope across different runs)")
        print(f"{'-'*70}")
        repeated = fabricated.groupby(
            fabricated["evidence_slope"].round(4)
        ).filter(lambda g: len(g) >= 2)
        if len(repeated) > 0:
            print(repeated[["chunk_id", "model", "evidence_slope", "ground_truth_slope"]]
                  .sort_values("evidence_slope").to_string(index=False))
        else:
            print("(none found — fabricated values are not obviously repeated placeholders)")

    # ---- Verdict ----
    print(f"\n{'='*70}")
    overall_problem_pct = df["issue_any"].sum() / total * 100
    if overall_problem_pct > 10:
        print(f"VERDICT: {overall_problem_pct:.1f}% of rows have issues. "
              f"NOT recommended to score as-is.")
        print("Recommend: fix the grounding issue in the pipeline that generated this data, "
              "or explicitly exclude flagged rows before computing accuracy metrics.")
    else:
        print(f"VERDICT: {overall_problem_pct:.1f}% of rows have issues — relatively clean, "
              f"but still recommend excluding flagged rows from scoring.")
    print(f"{'='*70}\n")

    return df


def main():
    parser = argparse.ArgumentParser(description="Check LLM results CSV for data quality issues before scoring.")
    parser.add_argument("csv_path", help="Path to the results CSV file")
    parser.add_argument("--tolerance", type=float, default=0.5,
                         help="Max allowed difference between cited and ground-truth slope "
                              "before flagging as fabricated evidence (default: 0.5)")
    parser.add_argument("--save-flagged", action="store_true",
                         help="Save a copy of the CSV with issue columns added, for inspection")
    args = parser.parse_args()

    df = check_results(args.csv_path, tolerance=args.tolerance)

    if args.save_flagged:
        out_path = args.csv_path.replace(".csv", "_flagged.csv")
        df.to_csv(out_path, index=False)
        print(f"Flagged CSV saved to: {out_path}")


if __name__ == "__main__":
    main()
