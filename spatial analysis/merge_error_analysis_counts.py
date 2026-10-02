"""
merge_error_analysis_counts.py
=================================
Reads every "error_analysis_<task>_counts.csv" file from each of the 4
configuration folders (Errors/single turn, Errors/multi turn,
Errors/ReAct, Errors/multi agent), merges them, and writes TWO output
files:
  1. error_analysis_merged_by_task.csv   -- one row per (config, task),
     raw counts per error_type.
  2. error_analysis_merged_by_config.csv -- one row per config, counts
     pooled across all 5 tasks. This is the input for the 100%
     stacked bar chart.

Each counts CSV only has a column for an error_type if that type
actually occurred at least once in that (task, mode) combination -- a
task with zero "ungrounded_correctness" cases has no such column at
all. Missing columns are filled with 0 after merging (absence of the
column means zero cases, not missing data) -- this exact case is
covered by the test in this script's accompanying message.

Usage:
    python merge_error_analysis_counts.py
"""

import glob
import os
import pandas as pd

ERROR_TYPES = ["format_failure", "grounded_correctness", "grounding_failure",
               "reasoning_failure", "ungrounded_correctness"]

# Folder path -> the config label you want in the final plot/paper.
# Edit the folder paths on the left if your actual directories differ.
CONFIG_DIRS = {
    "Errors/single turn": "Single agent single turn",
    "Errors/multi turn": "Single agent multi turn",
    "Errors/ReAct": "ReAct baseline",
    "Errors/multi agent": "SpatialInsightAgent",
}


def run():
    all_frames = []
    for dir_path, config_label in CONFIG_DIRS.items():
        files = glob.glob(os.path.join(dir_path, "*_counts.csv"))
        if not files:
            print(f"WARNING: no *_counts.csv files found in '{dir_path}' -- skipping.")
            continue
        print(f"{dir_path}: found {len(files)} file(s): "
              f"{', '.join(os.path.basename(f) for f in files)}")
        for f in files:
            d = pd.read_csv(f)
            d["config"] = config_label
            all_frames.append(d)

    if not all_frames:
        raise SystemExit(
            "No counts files found in any configuration folder -- check that "
            "CONFIG_DIRS paths at the top of this script match your actual "
            "Errors/ subfolder names."
        )

    all_counts = pd.concat(all_frames, ignore_index=True)

    # Fill in any error_type columns that were entirely absent from a
    # given file (zero cases of that type -> no column at all). Kept as
    # FLOAT here, not rounded/cast to int -- bivariate/outlier_r/
    # changespillover's values are already whole numbers (0/1 per
    # case), but getisord/spatial's are genuine fractional partial
    # credit (the same F1 the accuracy table uses), and rounding this
    # early would truncate that precision before it's ever summed.
    for col in ERROR_TYPES:
        if col not in all_counts.columns:
            all_counts[col] = 0.0
    all_counts[ERROR_TYPES] = all_counts[ERROR_TYPES].fillna(0.0)

    by_task = all_counts[["config", "task", "mode"] + ERROR_TYPES]
    by_task.to_csv("error_analysis_merged_by_task.csv", index=False)
    print(f"\nSaved: error_analysis_merged_by_task.csv ({len(by_task)} rows, "
          f"UNROUNDED -- getisord/spatial rows carry genuine fractional credit)")

    # --- Round ONLY here, on the final total pooled across all 5 tasks ---
    # This is deliberately the one and only rounding step in the whole
    # pipeline: summing exact fractional credit first and rounding the
    # total once minimizes the gap against the accuracy table (verified
    # numerically -- roughly 50x smaller gap than rounding each case
    # individually with a 0.5 threshold). round-half-up is used
    # explicitly rather than Python's default round() (which rounds
    # .5 to the nearest EVEN integer, "banker's rounding" -- a
    # surprising, easy-to-miss behavior for a reader checking the math
    # by hand); all totals here are non-negative, so this is safe.
    def _round_half_up(x):
        import math
        return int(math.floor(x + 0.5))

    by_config_fractional = all_counts.groupby("config")[ERROR_TYPES].sum().reset_index()
    by_config = by_config_fractional.copy()
    for col in ERROR_TYPES:
        by_config[col] = by_config_fractional[col].apply(_round_half_up)

    by_config.to_csv("error_analysis_merged_by_config.csv", index=False)
    print(f"Saved: error_analysis_merged_by_config.csv ({len(by_config)} rows, "
          f"DISCRETE -- rounded once, at this final pooled total)")
    print()
    print("Unrounded pooled totals (for reference):")
    print(by_config_fractional.to_string(index=False))
    print()
    print("Final discrete totals (this is the stacked-bar input):")
    print(by_config.to_string(index=False))


if __name__ == "__main__":
    run()
