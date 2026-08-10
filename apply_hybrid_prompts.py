"""
apply_hybrid_prompts.py
==========================
Takes your UPLOADED manifest.csv (the pre-casual, formula-heavy version)
as the base and makes ONE surgical change: rewrites insight_def to the
HYBRID format -- casual plain-English wording, but the FINAL ANSWER line
must include a cited computed number, e.g.:
    FINAL ANSWER: CLUSTERED (Moran's I = 0.17, p = 0.006)
    FINAL ANSWER: Camden (HH, I=1.04), Westminster (HH, I=1.34)

Everything else in the manifest -- ground_truth_present, ground_truth_detail,
ground_truth_rule, ground_truth_metric_json, csv_path, n_rows, window_*,
adjacency_path, dataset, modality, insight_type, spatial_granularity -- is
copied through UNCHANGED from your uploaded file. No ground truth is
recomputed or altered.

Adds one new column: answer_format="casual_with_citation", which tells
single_agent.py/multi_agent.py to use casual_verdict.parse_casual_with_citation()
instead of either the pure-JSON or pure-casual parser.

Usage:
    python apply_hybrid_prompts.py
Input:  manifest.csv (your uploaded pre-casual version, place in this folder)
Output: chunks_csv/manifest.csv (rewritten, 74 rows, same order)
"""

import pandas as pd
import re

INPUT_MANIFEST = "manifest.csv"          # your uploaded file
OUTPUT_MANIFEST = "chunks_csv/manifest.csv"

INDICATOR_LABELS = {
    "crime": "crime rate", "waste": "waste generation", "no2": "NO2 air pollution",
    "ghg": "greenhouse gas emissions", "gas": "gas consumption",
    "energy": "energy consumption", "transport": "transport energy use",
}

SELFCONTAINMENT_LABELS = {
    "selfcontainmentratio": "Self-Containment Ratio",
    "selfcontainmentindex": "Self-Containment Index",
}

SELFCONTAINMENT_FORMULAS = {
    "selfcontainmentratio": """1. Departures = number of trips that started in that borough
2. Self-contained trips = number of those departures that also ended in the same borough
3. Self-Containment Ratio = Self-contained trips / Departures""",
    "selfcontainmentindex": """1. Departures = number of trips that started in that borough
2. Self-contained trips = number of those departures that also ended in the same borough
3. Arrival share = (number of trips that ended in that borough) / (total trips that ended anywhere, across all boroughs)
4. Expected self-contained trips = Departures x Arrival share (this is what you'd expect by pure chance if destinations were random, not influenced by where the trip started)
5. Self-Containment Index = Self-contained trips / Expected self-contained trips""",
}

NO_GEOSPATIAL_LIBS_NOTICE = (
    "Note: dedicated spatial-statistics packages (pysal, esda, libpysal, geopandas, or similar) "
    "are NOT installed in this environment -- don't spend turns trying different import paths "
    "for them. Implement Moran's I directly using only pandas and numpy."
)

PERMUTATION_INSTRUCTION = (
    "If you test statistical significance via a permutation/randomization test, use at least "
    "500 random shuffles -- fewer permutations can produce an unstable p-value that lands on "
    "the wrong side of the 0.05 threshold just by chance, especially when the true effect is "
    "only moderate."
)

HYBRID_GLOBAL_MORAN = """I've attached {label} data for 33 London boroughs and a list showing which boroughs are geographic neighbors of each other.

Question: Overall across the whole city, do neighboring boroughs tend to have similar {label} (high values located next to other high values, and low next to low), or does {label} appear randomly distributed with no relation to which boroughs are next to each other?

Explain your reasoning based on the data, including the actual computed statistic (e.g. Global Moran's I and its significance) that supports your conclusion. """ + NO_GEOSPATIAL_LIBS_NOTICE + " " + PERMUTATION_INSTRUCTION + """

End your response with exactly one line, including the computed number(s) you used:
FINAL ANSWER: CLUSTERED (Moran's I = <value>, p = <value>)
or
FINAL ANSWER: NOT CLUSTERED (Moran's I = <value>, p = <value>)"""

HYBRID_LOCAL_MORAN = """I've attached the same {label} data for 33 London boroughs and the same borough adjacency list.

This time, I want specific individual boroughs (not an overall city-wide pattern) that fall into one of these two categories:
- HOT SPOT: a borough with a high {label} that is ALSO surrounded by other high-{label} neighboring boroughs
- COLD SPOT: a borough with a low {label} that is ALSO surrounded by other low-{label} neighboring boroughs

Look at each borough together with its neighbors from the adjacency list, and identify any that qualify. Explain your reasoning, including a computed value for each borough you name. """ + NO_GEOSPATIAL_LIBS_NOTICE + " " + PERMUTATION_INSTRUCTION + """

End your response with exactly one line listing every borough you identified, using HH for hot spot and LL for cold spot, and including a computed value for each:
FINAL ANSWER: BoroughName1 (HH, I=<value>), BoroughName2 (LL, I=<value>), ...

If you find none, write: FINAL ANSWER: none"""

HYBRID_GLOBAL_SELFCONTAINMENT = """I've attached trip-level bike-share data (columns include StartBorough, EndBorough) and a borough adjacency list.

First, compute a "{metric_label}" for each borough, using exactly this method:
{formula}

Now, using the borough adjacency list: overall across the whole area, do neighboring boroughs tend to have similar {metric_label} values (high values next to other high values, low next to low), or does the index appear randomly distributed with no relation to which boroughs are next to each other?

Explain your reasoning based on the data, including the actual computed statistic (e.g. Global Moran's I and its significance) that supports your conclusion. """ + NO_GEOSPATIAL_LIBS_NOTICE + " " + PERMUTATION_INSTRUCTION + """

End your response with exactly one line, including the computed number(s) you used:
FINAL ANSWER: CLUSTERED (Moran's I = <value>, p = <value>)
or
FINAL ANSWER: NOT CLUSTERED (Moran's I = <value>, p = <value>)"""

HYBRID_LOCAL_SELFCONTAINMENT = """I've attached the same trip-level bike-share data and the same borough adjacency list.

First, compute a "{metric_label}" for each borough, using exactly this method:
{formula}

Now, using the borough adjacency list, identify specific individual boroughs (not an overall city-wide pattern) that fall into one of these two categories:
- HOT SPOT: a borough with a high {metric_label} that is ALSO surrounded by other high-{metric_label} neighboring boroughs
- COLD SPOT: a borough with a low {metric_label} that is ALSO surrounded by other low-{metric_label} neighboring boroughs

Explain your reasoning, including a computed value for each borough you name. """ + NO_GEOSPATIAL_LIBS_NOTICE + " " + PERMUTATION_INSTRUCTION + """

End your response with exactly one line listing every borough you identified, using HH for hot spot and LL for cold spot, and including a computed value for each:
FINAL ANSWER: BoroughName1 (HH, I=<value>), BoroughName2 (LL, I=<value>), ...

If you find none, write: FINAL ANSWER: none"""


def build_insight_def(chunk_id):
    m = re.match(r"moran_(global|local)_\d{4}_(\w+)$", chunk_id)
    if m:
        granularity, indicator_key = m.group(1), m.group(2)
        label = INDICATOR_LABELS.get(indicator_key, indicator_key)
        return HYBRID_GLOBAL_MORAN.format(label=label) if granularity == "global" \
            else HYBRID_LOCAL_MORAN.format(label=label)

    m = re.match(r"cyclehire_(global|local)_(\w+?)_w\d{3}$", chunk_id)
    if m:
        granularity, metric_key = m.group(1), m.group(2)
        metric_label = SELFCONTAINMENT_LABELS.get(metric_key, metric_key)
        formula = SELFCONTAINMENT_FORMULAS.get(metric_key, "")
        return HYBRID_GLOBAL_SELFCONTAINMENT.format(metric_label=metric_label, formula=formula) \
            if granularity == "global" \
            else HYBRID_LOCAL_SELFCONTAINMENT.format(metric_label=metric_label, formula=formula)

    raise ValueError(f"Unrecognized chunk_id pattern: {chunk_id}")


def run():
    df = pd.read_csv(INPUT_MANIFEST)
    print(f"Loaded {len(df)} rows from {INPUT_MANIFEST} (ground truth columns untouched)")

    df["insight_def"] = df["chunk_id"].apply(build_insight_def)
    df["answer_format"] = "casual_with_citation"

    df.to_csv(OUTPUT_MANIFEST, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_MANIFEST}")
    print(f"Sample new insight_def (moran_global_2023_gas):")
    sample = df[df.chunk_id == "moran_global_2023_gas"]
    if len(sample) > 0:
        print(sample.iloc[0]["insight_def"])


if __name__ == "__main__":
    run()
