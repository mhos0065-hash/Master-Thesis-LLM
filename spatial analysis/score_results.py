"""
score_results.py
==================
Evaluates results.csv (produced by single_agent.py / multi_agent.py /
run_experiment.py) against the withheld ground truth in manifest.csv,
computing precision/recall/F1 disaggregated by pipeline mode, model,
insight type, AND modality (spatial vs non-spatial) -- Section 4.7.

Also runs McNemar's test comparing single-agent vs multi-agent correctness
on matched chunks (same chunk_id, same model) -- Section 4.7: "McNemar's
test will assess whether differences in correctness between pipeline
configurations are statistically significant."

`predicted` now comes directly from the Pydantic-validated verdict's
insight_present field (see schemas.py / agent_utils.py) rather than
text-matching "FOUND"/"NOT FOUND" -- a row with an empty/ambiguous
`predicted` means the agent never produced a schema-valid, evidence-citing
verdict, and is scored as incorrect, never silently dropped.

Usage:
    python score_results.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

RESULTS_CSV = "results.csv"


def load():
    df = pd.read_csv(RESULTS_CSV)
    df["predicted_bool"] = df["predicted"].map({"True": True, "False": False, True: True, False: False})
    df["ground_truth_bool"] = df["ground_truth"].astype(bool)
    df["ambiguous"] = df["predicted_bool"].isna()
    df["correct"] = (~df["ambiguous"]) & (df["predicted_bool"] == df["ground_truth_bool"])
    return df


def report(df, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols):
        if len(g) == 0:
            continue
        # ambiguous rows (no valid verdict) are treated as an effective
        # "False" prediction for precision/recall purposes -- the model made
        # no positive claim, so it can never count as a false positive, but
        # it DOES count as a miss (false negative) when the real answer was
        # True. Accuracy's denominator is the FULL count, not just the rows
        # with a valid verdict -- an ambiguous row is never "correct" just
        # because it happens to match an all-False default.
        y_true = g["ground_truth_bool"].astype(int)
        effective_pred = g["predicted_bool"].fillna(False).astype(int)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["n"] = len(g)
        row["n_ambiguous"] = int(g["ambiguous"].sum())
        row["pct_with_evidence"] = round(g["has_evidence"].fillna(False).mean(), 3) if "has_evidence" in g else None
        row["accuracy"] = round(g["correct"].sum() / len(g), 3)
        row["precision"] = round(precision_score(y_true, effective_pred, zero_division=0), 3)
        row["recall"] = round(recall_score(y_true, effective_pred, zero_division=0), 3)
        row["f1"] = round(f1_score(y_true, effective_pred, zero_division=0), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def mcnemar_single_vs_multi(df):
    """Section 4.7: McNemar's test on matched chunk_id + model pairs, comparing
    whether single-agent and multi-agent get the SAME chunk right or wrong.
    Requires both modes to have a result for the same (chunk_id, model)."""
    pivot = df.pivot_table(index=["chunk_id", "model"], columns="mode",
                            values="correct", aggfunc="first")
    if "single" not in pivot.columns or "multi" not in pivot.columns:
        return None, "Need both single-agent and multi-agent results for the same chunks/models to run McNemar's test."

    matched = pivot.dropna(subset=["single", "multi"])
    if len(matched) < 4:
        return None, f"Only {len(matched)} matched (chunk_id, model) pairs with both modes present -- " \
                      f"McNemar's test needs a reasonable sample. Run more chunks/models through both pipelines."

    # 2x2 contingency table: [[both correct, single correct only], [multi correct only, both wrong]]
    both_correct = int(((matched["single"] == True) & (matched["multi"] == True)).sum())
    single_only = int(((matched["single"] == True) & (matched["multi"] == False)).sum())
    multi_only = int(((matched["single"] == False) & (matched["multi"] == True)).sum())
    both_wrong = int(((matched["single"] == False) & (matched["multi"] == False)).sum())
    table = [[both_correct, single_only], [multi_only, both_wrong]]

    result = mcnemar(table, exact=(single_only + multi_only) < 25, correction=True)
    return {
        "n_matched_pairs": len(matched),
        "contingency_table": table,
        "single_correct_multi_wrong": single_only,
        "multi_correct_single_wrong": multi_only,
        "statistic": round(float(result.statistic), 4),
        "p_value": round(float(result.pvalue), 4),
        "significant_at_0.05": bool(result.pvalue < 0.05),
    }, None


if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} results ({df['ambiguous'].sum()} ambiguous/no-valid-verdict rows)\n")

    print("=== By pipeline mode (single vs multi) ===")
    print(report(df, ["mode"]).to_string(index=False))

    print("\n=== By model ===")
    print(report(df, ["model"]).to_string(index=False))

    print("\n=== By pipeline mode x model ===")
    print(report(df, ["mode", "model"]).to_string(index=False))

    print("\n=== By insight type ===")
    print(report(df, ["insight_type"]).to_string(index=False))

    print("\n=== By modality (spatial vs non-spatial) ===")
    if "modality" in df.columns:
        print(report(df, ["modality"]).to_string(index=False))
    else:
        print("(no 'modality' column in results.csv -- regenerate chunks with the updated generate_all_chunks.py)")

    print("\n=== By insight type x mode ===")
    print(report(df, ["insight_type", "mode"]).to_string(index=False))

    print("\n=== McNemar's test: single-agent vs multi-agent (Section 4.7) ===")
    mcnemar_result, err = mcnemar_single_vs_multi(df)
    if mcnemar_result:
        for k, v in mcnemar_result.items():
            print(f"  {k}: {v}")
        if mcnemar_result["significant_at_0.05"]:
            print("  -> Statistically significant difference between single- and multi-agent correctness (p<0.05).")
        else:
            print("  -> No statistically significant difference detected at alpha=0.05 "
                  "(may just mean you need more matched chunks/models before drawing conclusions).")
    else:
        print(f"  Not run: {err}")

    out_path = "scoring_summary.csv"
    report(df, ["mode", "model", "insight_type", "modality"] if "modality" in df.columns
           else ["mode", "model", "insight_type"]).to_csv(out_path, index=False)
    print(f"\nFull breakdown saved to {out_path}")
