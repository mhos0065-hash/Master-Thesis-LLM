"""
generate_all_chunks.py
=======================
Builds ONE small CSV per QuickInsights insight type (minimum viable set for
your first end-to-end run), each drawn from a different one of your six
London datasets, plus a manifest.json holding the ground truth + insight
definition for each chunk. Ground truth is NEVER written into the CSVs
themselves -- it is looked up only at scoring time (score_results.py).

Run once:
    python3 generate_all_chunks.py

Outputs go to:  chunks_csv/*.csv   (what the Coder agent loads)
                chunks_csv/manifest.json  (ground truth + insight_def, scoring only)

NOTE: sample_*.csv inputs here are small hand-built extracts from the rows
you pasted, for pipeline testing only. One row in sample_cyclehire.csv
(RentalId 50754243, Duration 4980s) was deliberately added as a clear
outlier so the 'outlier' chunk has a genuine positive case to test against
-- replace with your real journey extract before running actual experiments.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

from manifest_io import save_manifest

CSV_DIR = "/mnt/user-data/outputs/chunks_csv"
os.makedirs(CSV_DIR, exist_ok=True)
MANIFEST_PATH = os.path.join(CSV_DIR, "manifest.csv")

# ---------------------------------------------------------------------
# Ground-truth statistical tests -- one per insight type.
# These mirror the QuickInsights-style "mathematical acceptance criteria"
# from Section 4.3/4.7 of the proposal. Thresholds are defensible defaults;
# calibrate them against your human annotators' Cohen's-Kappa-agreed labels.
# ---------------------------------------------------------------------

def gt_outlier(labels, values, z_thresh=3.0):
    v = np.array(values, dtype=float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) or 1e-9
    mz = 0.6745 * (v - med) / mad
    flagged = [labels[i] for i in range(len(v)) if abs(mz[i]) > z_thresh]
    return {"insight_present": len(flagged) > 0,
            "verdict_detail": f"{len(flagged)} outlier(s)" + (f": {flagged}" if flagged else ""),
            "metric": {"modified_z_scores": dict(zip(labels, np.round(mz, 2).tolist())), "threshold": z_thresh},
            "acceptance_rule": "flag if |0.6745*(x-median)/MAD| > 3.0"}

def gt_outstanding_value(labels, values, ratio_thresh=2.0):
    v = np.array(values, dtype=float)
    order = np.argsort(-v)
    top, second = v[order[0]], v[order[1]] if len(v) > 1 else 0
    ratio = top / second if second not in (0, np.nan) else np.inf
    present = ratio >= ratio_thresh
    return {"insight_present": bool(present),
            "verdict_detail": f"top '{labels[order[0]]}'={top:.2f} is {ratio:.2f}x second '{labels[order[1]]}'={second:.2f}",
            "metric": {"top_label": labels[order[0]], "top_value": float(top),
                       "second_label": labels[order[1]], "second_value": float(second), "ratio": float(ratio)},
            "acceptance_rule": f"present if top/second ratio >= {ratio_thresh}"}

def gt_trend(values, alpha=0.05):
    v = np.array(values, dtype=float)
    mask = ~np.isnan(v)
    t = np.arange(len(v))[mask]
    v = v[mask]
    if len(v) < 4:
        return {"insight_present": False, "verdict_detail": "insufficient points", "metric": {}, "acceptance_rule": "n>=4"}
    rho, p = stats.spearmanr(t, v)
    present = p < alpha
    direction = "increasing" if rho > 0 else "decreasing"
    return {"insight_present": bool(present),
            "verdict_detail": f"{'significant' if present else 'no significant'} {direction} trend (rho={rho:.2f}, p={p:.3f})",
            "metric": {"spearman_rho": round(float(rho), 3), "p_value": round(float(p), 4)},
            "acceptance_rule": f"present if p < {alpha}"}

def gt_seasonality(values, period=6, alpha=0.05):
    v = np.array(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n <= period:
        return {"insight_present": False, "verdict_detail": "insufficient points", "metric": {}, "acceptance_rule": f"n>{period}"}
    v_c = v - v.mean()
    acf = np.correlate(v_c, v_c, mode="full")[n-1:] / (np.var(v) * n)
    acf_at_lag = acf[period] if period < len(acf) else np.nan
    ci = 1.96 / np.sqrt(n)
    present = abs(acf_at_lag) > ci
    return {"insight_present": bool(present),
            "verdict_detail": f"ACF at lag {period}={acf_at_lag:.2f} ({'exceeds' if present else 'within'} CI ±{ci:.2f})",
            "metric": {"acf_lag": round(float(acf_at_lag), 3), "ci_95": round(float(ci), 3)},
            "acceptance_rule": "present if |ACF(period)| > 1.96/sqrt(n)"}

def gt_evenness(values):
    v = np.sort(np.array(values, dtype=float))
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0 or v.sum() == 0:
        return {"insight_present": False, "verdict_detail": "no data", "metric": {}, "acceptance_rule": "n/a"}
    cum = np.cumsum(v)
    gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    uneven = gini > 0.3
    return {"insight_present": bool(uneven),
            "verdict_detail": f"Gini={gini:.2f} -> {'uneven/concentrated' if uneven else 'roughly even'}",
            "metric": {"gini_coefficient": round(float(gini), 3)},
            "acceptance_rule": "present ('uneven') if Gini > 0.30"}

def gt_attribution(df, group_col, value_col, top_n=1, share_thresh=0.5):
    g = df.groupby(group_col)[value_col].sum().sort_values(ascending=False)
    share = g / g.sum()
    dominant = share.head(top_n)
    present = dominant.sum() > share_thresh
    return {"insight_present": bool(present),
            "verdict_detail": f"{list(dominant.index)} = {dominant.sum()*100:.1f}% of total {value_col}",
            "metric": {"share_by_category": {k: round(float(x), 3) for k, x in share.items()}},
            "acceptance_rule": f"present if top {top_n} contributor(s) explain > {share_thresh*100:.0f}% of total"}

def gt_spatial_correlation(x, y, alpha=0.05):
    """NOTE: no longer used to build spatial_correlation chunks -- this tests
    plain bivariate Pearson correlation between two DIFFERENT variables,
    which is NOT the same statistical concept as spatial autocorrelation
    (Moran's I). The real spatial_correlation chunks now use
    GLOBAL_MORAN_GROUND_TRUTH + build_moran_chunks() below, with ground
    truth from your actual R analysis. Kept here only in case you want a
    plain cross-variable-correlation insight type for some other purpose --
    it is not called anywhere in build()."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return {"insight_present": False, "verdict_detail": "insufficient paired points", "metric": {}, "acceptance_rule": "n>=4"}
    r, p = stats.pearsonr(x, y)
    present = p < alpha and abs(r) > 0.3
    return {"insight_present": bool(present),
            "verdict_detail": f"Pearson r={r:.2f} (p={p:.3f})",
            "metric": {"pearson_r": round(float(r), 3), "p_value": round(float(p), 4)},
            "acceptance_rule": f"present if p<{alpha} and |r|>0.3"}

# ---------------------------------------------------------------------
# Insight definitions -- text that goes INTO the prompt (ground truth does not)
# ---------------------------------------------------------------------

INSIGHT_DEFS = {
    "outlier": "Is there a statistical outlier among the rows in this dataset? "
               "Ground your answer in code: compute a robust z-score (or IQR) per row/unit. "
               "State FOUND or NOT FOUND, name the unit(s), and cite the computed statistic.",
    "outstanding_value": "Does one row/unit stand out as dominant compared to the rest? "
               "Compute the top value vs. the rest (e.g. ratio to second-highest). "
               "State FOUND or NOT FOUND and cite the computed values.",
    "trend": "Is there a statistically meaningful increasing or decreasing trend over time? "
               "Compute a trend statistic (e.g. correlation or slope with time, and a significance measure). "
               "State FOUND or NOT FOUND, direction, and cite the computed statistic.",
    "seasonality": "Is there a recurring seasonal/cyclic pattern in this time series? "
               "Compute an autocorrelation or seasonal-decomposition based statistic. "
               "State FOUND or NOT FOUND and cite the computed statistic.",
    "evenness": "Is the distribution of values across units notably uneven/concentrated, or roughly even? "
               "Compute a concentration statistic (e.g. Gini coefficient). "
               "State FOUND ('uneven') or NOT FOUND ('even') and cite the computed statistic.",
    "attribution": "Can the total be attributed mainly to one or a few categories/columns? "
               "Compute each category's share of the total. "
               "State FOUND or NOT FOUND, name the dominant category(ies), and cite the computed shares.",
    "spatial_correlation": "Does this indicator show significant spatial autocorrelation (clustering) "
               "across geographic units, given their adjacency structure? "
               "Compute Moran's I (or an equivalent spatial-lag-based statistic) using the provided "
               "adjacency weights. State FOUND or NOT FOUND, and cite the computed statistic and its "
               "significance as evidence. (Default definition -- see chunk-specific overrides for "
               "Global vs Local Moran's I chunks, which give the exact formula and files to use.)",
}

GLOBAL_MORAN_INSIGHT_DEF = (
    "Does this indicator show significant GLOBAL spatial autocorrelation (clustering) across London "
    "boroughs? You are given the indicator's value per borough (DATASET_PATH) AND a borough adjacency "
    "list (ADJACENCY_PATH, columns: borough, neighbor -- each row means these two boroughs are "
    "geographic neighbors; the list is symmetric). Replace any missing values with the column mean "
    "first. Compute Global Moran's I: "
    "I = (n/S0) * [sum_i sum_j w_ij*(x_i-xbar)*(x_j-xbar)] / [sum_i (x_i-xbar)^2], "
    "using row-standardized weights w_ij = 1/(number of neighbors of borough i), and S0 = sum of all "
    "w_ij across every (i,j) pair in the adjacency list. Then estimate significance via permutation: "
    "shuffle the indicator values across boroughs (keep the adjacency list the same) at least 200 "
    "times, recompute I for each shuffle, and estimate a p-value as the fraction of shuffled I values "
    "that are >= your observed I (this mirrors a Monte Carlo Moran's I test). State FOUND (significant "
    "positive clustering, e.g. permutation p < 0.05) or NOT FOUND, and cite your computed I and "
    "permutation p-value as evidence."
)

LOCAL_MORAN_INSIGHT_DEF = (
    "Does at least one borough form a significant LOCAL spatial cluster (LISA) for this indicator? "
    "You are given the indicator's value per borough (DATASET_PATH) AND a borough adjacency list "
    "(ADJACENCY_PATH). Replace missing values with the column mean, then standardize values to "
    "z-scores. For each borough i, compute its spatial lag = the average z-score of its neighbors "
    "(row-standardized weights from the adjacency list). Classify each borough: HH if z_i>0 and its "
    "lag>0 (high value surrounded by high values); LL if z_i<0 and lag<0 (low surrounded by low); "
    "HL/LH if they disagree (spatial outliers, not clusters). A borough counts as a significant "
    "cluster if it is HH or LL AND its local statistic (z_i * lag_i) is unusually extreme relative to "
    "a permutation-based null (shuffle values across boroughs many times, recompute z_i*lag_i each "
    "time, and treat the observed value as significant if it falls in the extreme ~10% tail). State "
    "FOUND if at least one borough qualifies as a significant HH or LL cluster, name the borough(s), "
    "and cite your computed z-score/lag/classification as evidence."
)

_manifest = {}

def save_chunk(chunk_id, dataset, insight_type, df, ground_truth, modality="non_spatial",
                window_start="", window_end="", window_index="", adjacency_path="",
                insight_def_override=None, spatial_granularity=""):
    csv_path = os.path.join(CSV_DIR, f"{chunk_id}.csv")
    df.to_csv(csv_path, index=False)
    _manifest[chunk_id] = {
        "dataset": dataset,
        "insight_type": insight_type,
        "modality": modality,
        "csv_path": f"chunks_csv/{chunk_id}.csv",
        "n_rows": len(df),
        "insight_def": insight_def_override or INSIGHT_DEFS[insight_type],
        "ground_truth": ground_truth,
        "window_start": window_start,
        "window_end": window_end,
        "window_index": window_index,
        "adjacency_path": adjacency_path,
        "spatial_granularity": spatial_granularity,
    }
    print(f"[{insight_type:>18}] {chunk_id:<40} present={ground_truth['insight_present']!s:<5} "
          f"modality={modality:<11} | {ground_truth['verdict_detail']}")

# ---------------------------------------------------------------------
# Build the 7 chunks
# ---------------------------------------------------------------------

def build():
    # 1) OUTLIER -- Santander Cycle Hire journey durations (single-location distribution -> non-spatial)
    cyc = pd.read_csv("/home/claude/sample_cyclehire.csv")
    gt = gt_outlier(cyc["RentalId"].astype(str).tolist(), cyc["Duration"].tolist())
    save_chunk("cyclehire_outlier_sample", "Santander Cycle Hire", "outlier", cyc, gt, modality="non_spatial")

    # 2) OUTSTANDING VALUE -- Gas consumption, avg commercial consumption ACROSS BOROUGHS -> spatial
    gas = pd.read_csv("/home/claude/sample_gas.csv")
    gt = gt_outstanding_value(gas["Area"].tolist(), gas["AvgCommercialConsumptionKWh"].tolist())
    save_chunk("gas_outstanding_sample", "Gas Consumption by Borough", "outstanding_value", gas, gt, modality="spatial")

    # 3) TREND -- MPS Crime, monthly CRIMINAL DAMAGE counts, ONE ward over time -> non-spatial
    crime_t = pd.read_csv("/home/claude/sample_crime_trend.csv")
    cd = crime_t[crime_t["MinorText"] == "CRIMINAL DAMAGE"].sort_values("Month")
    gt = gt_trend(cd["Count"].tolist())
    save_chunk("crime_trend_heathrowvillages", "MPS Recorded Crime", "trend", cd, gt, modality="non_spatial")

    # 4) SEASONALITY -- NO2 diffusion tube, ONE site's monthly series -> non-spatial
    no2 = pd.read_csv("/home/claude/sample_no2.csv")
    MONTH_COLS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sept","Oct","Nov","Dec"]
    no2[MONTH_COLS] = no2[MONTH_COLS].apply(pd.to_numeric, errors="coerce")
    site = no2.iloc[4]  # Boys School Sports Access Ramp -- fullest series (100% capture)
    monthly_df = pd.DataFrame({"Month": MONTH_COLS, "NO2_ugm3": site[MONTH_COLS].values})
    gt = gt_seasonality(site[MONTH_COLS].values, period=6)
    save_chunk("no2_seasonality_boysschool", "NO2 Diffusion Tube", "seasonality", monthly_df, gt, modality="non_spatial")

    # 5) EVENNESS -- MPS Crime, crime-TYPE distribution for one ward-month -> non-spatial
    crime_e = pd.read_csv("/home/claude/sample_crime_evenness.csv")
    gt = gt_evenness(crime_e["Count_202603"].tolist())
    save_chunk("crime_evenness_heathrowvillages_202603", "MPS Recorded Crime", "evenness", crime_e, gt, modality="non_spatial")

    # 6) ATTRIBUTION -- LEGGI, SECTOR breakdown for one borough -> non-spatial
    leggi = pd.read_csv("/home/claude/sample_leggi.csv")
    city = leggi[leggi["Borough"] == "City of London"].iloc[0]
    sector_df = pd.DataFrame({
        "Sector": ["Domestic", "IndustrialCommercial", "Transport", "NRMM"],
        "EmissionsTotal": [city["Domestic_Total"], city["IndustrialCommercial_Total"],
                            city["Transport_Total"], city["NRMM_Total"]],
    })
    gt = gt_attribution(sector_df, "Sector", "EmissionsTotal", top_n=1)
    save_chunk("leggi_attribution_cityoflondon", "LEGGI", "attribution", sector_df, gt, modality="non_spatial")

    # 7) SPATIAL CORRELATION -- Global + Local Moran's I, real R-validated ground truth
    #    (see build_moran_chunks() below -- this REPLACES an earlier version that used
    #    plain bivariate correlation between two DIFFERENT variables across boroughs,
    #    which is NOT the same statistical concept as spatial autocorrelation and was
    #    removed to avoid two different ground-truth definitions sharing one insight_type)
    build_moran_chunks()

    # bonus: evenness ACROSS BOROUGHS' LEGGI grand totals -> spatial (2nd evenness example, different modality than #5)
    gt = gt_evenness(leggi["GrandTotal"].tolist())
    save_chunk("leggi_evenness_boroughs", "LEGGI", "evenness", leggi[["Borough", "GrandTotal"]], gt, modality="spatial")

    build_streaming_demo()

    save_manifest(_manifest, MANIFEST_PATH)
    print(f"\nmanifest written: {MANIFEST_PATH}  ({len(_manifest)} chunks)")


# ---------------------------------------------------------------------
# Spatial correlation chunks -- REAL Global + Local Moran's I
# ---------------------------------------------------------------------
# Ground truth here is NOT computed by a gt_* function like the other insight
# types -- it comes directly from your own R analysis (spdep::moran.mc/.test
# and localmoran_perm), transcribed from your actual console output and
# uploaded local_moran_results_all_years.csv. This was independently
# verified: a from-scratch numpy reimplementation of the Global Moran's I
# formula, using ONLY the uploaded borough_adjacency.csv and the real 2023
# borough values, reproduced your R-computed I to 3-4 decimal places on all
# 7 indicators. That's strong evidence the adjacency list and the values are
# both correct and mutually consistent.
#
# Each chunk needs TWO files: the borough values (csv_path, one row per
# borough) and the shared adjacency list (adjacency_path, same file for
# every Moran chunk). single_agent.py/multi_agent.py read adjacency_path
# from the manifest and pass it to agent_loop() as
# extra_files={"ADJACENCY_PATH": ...}, so the Coder agent can load both and
# compute Moran's I itself in pandas/numpy (no geospatial package needed --
# the formula is a direct weighted sum, fully computable from the adjacency
# list alone).
#
# Currently only 2023 has real per-borough VALUES available (from the row
# data you pasted) -- ground truth exists for 2021/2022 too (transcribed
# from your R console output and the uploaded local results), but building
# those years' chunks needs PREP/2021.csv and PREP/2022.csv (the raw panel
# files your R script reads) to supply real values for the LLM to compute on.

GLOBAL_MORAN_GROUND_TRUTH = {
    # (year, indicator) -> {moran_i, p_value_mc, p_value_std, significant, interpretation}
    (2021,"Crime"):{"moran_i":0.234,"p_value_mc":0.002,"p_value_std":0.0006,"significant":True,"interpretation":"Moderate clustering"},
    (2021,"Waste"):{"moran_i":0.185,"p_value_mc":0.037,"p_value_std":0.015,"significant":True,"interpretation":"Moderate clustering"},
    (2021,"Gas"):{"moran_i":-0.0072,"p_value_mc":0.098,"p_value_std":0.0964,"significant":False,"interpretation":"Dispersed/Random"},
    (2021,"Energy"):{"moran_i":-0.0118,"p_value_mc":0.136,"p_value_std":0.133,"significant":False,"interpretation":"Dispersed/Random"},
    (2021,"GHG"):{"moran_i":-0.0126,"p_value_mc":0.142,"p_value_std":0.144,"significant":False,"interpretation":"Dispersed/Random"},
    (2021,"NO2"):{"moran_i":-0.0455,"p_value_mc":0.515,"p_value_std":0.556,"significant":False,"interpretation":"Dispersed/Random"},
    (2021,"Transport"):{"moran_i":-0.0813,"p_value_mc":0.758,"p_value_std":0.772,"significant":False,"interpretation":"Dispersed/Random"},
    (2022,"Waste"):{"moran_i":0.169,"p_value_mc":0.03,"p_value_std":0.0235,"significant":True,"interpretation":"Moderate clustering"},
    (2022,"Crime"):{"moran_i":0.168,"p_value_mc":0.004,"p_value_std":0.001,"significant":True,"interpretation":"Moderate clustering"},
    (2022,"Gas"):{"moran_i":-0.005,"p_value_mc":0.09,"p_value_std":0.0816,"significant":False,"interpretation":"Dispersed/Random"},
    (2022,"Energy"):{"moran_i":-0.012,"p_value_mc":0.149,"p_value_std":0.143,"significant":False,"interpretation":"Dispersed/Random"},
    (2022,"GHG"):{"moran_i":-0.0128,"p_value_mc":0.153,"p_value_std":0.155,"significant":False,"interpretation":"Dispersed/Random"},
    (2022,"Transport"):{"moran_i":-0.0712,"p_value_mc":0.665,"p_value_std":0.706,"significant":False,"interpretation":"Dispersed/Random"},
    (2022,"NO2"):{"moran_i":-0.0847,"p_value_mc":0.714,"p_value_std":0.730,"significant":False,"interpretation":"Dispersed/Random"},
    (2023,"Crime"):{"moran_i":0.167,"p_value_mc":0.006,"p_value_std":0.0008,"significant":True,"interpretation":"Moderate clustering"},
    (2023,"Waste"):{"moran_i":0.104,"p_value_mc":0.097,"p_value_std":0.094,"significant":False,"interpretation":"Not significant"},
    (2023,"NO2"):{"moran_i":0.0837,"p_value_mc":0.136,"p_value_std":0.132,"significant":False,"interpretation":"Not significant"},
    (2023,"GHG"):{"moran_i":-0.0054,"p_value_mc":0.119,"p_value_std":0.101,"significant":False,"interpretation":"Dispersed/Random"},
    (2023,"Gas"):{"moran_i":-0.0088,"p_value_mc":0.101,"p_value_std":0.101,"significant":False,"interpretation":"Dispersed/Random"},
    (2023,"Energy"):{"moran_i":-0.0088,"p_value_mc":0.133,"p_value_std":0.121,"significant":False,"interpretation":"Dispersed/Random"},
    (2023,"Transport"):{"moran_i":-0.0709,"p_value_mc":0.682,"p_value_std":0.714,"significant":False,"interpretation":"Dispersed/Random"},
}

INDICATOR_TO_COLUMN = {
    "Crime": "crime_per_1k", "Waste": "waste_tonnes_per_capita", "NO2": "no2_annual_mean",
    "GHG": "ghg_tco2e_per_capita", "Gas": "gas_kwh_per_consumer", "Energy": "energy_kwh_per_capita",
    "Transport": "transport_kwh_per_capita",
}

YEARS_WITH_REAL_VALUES = {2023: "/home/claude/sample_moran_dataset_2023.csv"}
# Add more years here once you upload PREP/2021.csv / PREP/2022.csv, e.g.:
# YEARS_WITH_REAL_VALUES = {2021: "...", 2022: "...", 2023: "/home/claude/sample_moran_dataset_2023.csv"}


def build_moran_chunks():
    print("\n--- spatial correlation chunks (real Global + Local Moran's I) ---")
    ADJACENCY_SRC = "/home/claude/sample_borough_adjacency.csv"
    adjacency_df = pd.read_csv(ADJACENCY_SRC)

    # Save ONE shared adjacency file into chunks_csv/ -- every Moran chunk points at it
    adjacency_out_path = os.path.join(CSV_DIR, "borough_adjacency.csv")
    adjacency_df.to_csv(adjacency_out_path, index=False)
    adjacency_manifest_path = "chunks_csv/borough_adjacency.csv"

    local_df = pd.read_csv("/home/claude/sample_local_moran_results.csv")

    n_global, n_local = 0, 0
    for year, values_path in YEARS_WITH_REAL_VALUES.items():
        values = pd.read_csv(values_path)

        for indicator, col in INDICATOR_TO_COLUMN.items():
            value_df = values[["borough", col]].rename(columns={col: f"{indicator}_value"})

            # --- GLOBAL chunk ---
            gmt = GLOBAL_MORAN_GROUND_TRUTH.get((year, indicator))
            if gmt:
                # Defensive re-check here, not just trusting the pre-set "significant"
                # flag: a significant NEGATIVE Moran's I means significant dispersion,
                # not clustering, and must not count as insight_present=True. Matches
                # spatial_correlation_ground_truth.csv's rule exactly.
                is_sig = gmt["significant"] and gmt["moran_i"] > 0
                gt = {
                    "insight_present": is_sig,
                    "verdict_detail": f"Global Moran's I = {gmt['moran_i']:.4f} "
                                       f"(MC p-value={gmt['p_value_mc']}) -> {gmt['interpretation']}",
                    "metric": {"moran_i": gmt["moran_i"], "p_value_mc": gmt["p_value_mc"],
                               "p_value_std": gmt["p_value_std"]},
                    "acceptance_rule": "is_significant=True only if p_value (Monte Carlo, 999 perms) < 0.05 "
                                        "AND morans_i > 0 (a significant NEGATIVE I means significant dispersion, "
                                        "not clustering, and must not count as CLUSTERED)",
                }
                chunk_id = f"moran_global_{year}_{indicator.lower()}"
                save_chunk(chunk_id, f"Global Moran's I ({indicator}, {year})", "spatial_correlation",
                           value_df, gt, modality="spatial", adjacency_path=adjacency_manifest_path,
                           insight_def_override=GLOBAL_MORAN_INSIGHT_DEF, spatial_granularity="global")
                n_global += 1

            # --- LOCAL chunk (LISA) ---
            local_sub = local_df[(local_df["year"] == year) & (local_df["Indicator"] == indicator)]
            if len(local_sub) > 0:
                sig_clusters = local_sub[local_sub["cluster"].isin(["HH", "LL"])]
                present = len(sig_clusters) > 0
                cluster_list = [f"{r.borough} ({r.cluster})" for r in sig_clusters.itertuples()]
                # Full per-borough breakdown -- every borough, not just the
                # significant ones -- matching spatial_correlation_ground_truth.csv's
                # actual columns (z_score, lag_z, morans_i, p_value, cluster_label).
                per_borough = [
                    {"borough": r.borough, "z_score": float(r.z_score), "lag_z": float(r.lag_z),
                     "morans_i": float(r.Ii), "p_value": float(r.p_value), "cluster_label": r.cluster}
                    for r in local_sub.itertuples()
                ]
                gt = {
                    "insight_present": present,
                    "verdict_detail": (f"{len(sig_clusters)} borough(s) form significant local clusters: "
                                        f"{', '.join(cluster_list)}" if present else
                                        "No borough forms a significant HH or LL local cluster"),
                    "metric": {
                        "significant_cluster_boroughs": {r.borough: r.cluster for r in sig_clusters.itertuples()},
                        "cluster_counts": local_sub["cluster"].value_counts().to_dict(),
                        "per_borough": per_borough,
                    },
                    "acceptance_rule": "is_significant=True only if: (z_score>0 AND lag_z>0, i.e. HH) OR "
                                        "(z_score<0 AND lag_z<0, i.e. LL), AND p_value<0.10 (localmoran_perm()). "
                                        "cluster_label is not independent evidence -- it IS this sign(z_score)/"
                                        "sign(lag_z) test, restated as a label.",
                }
                chunk_id = f"moran_local_{year}_{indicator.lower()}"
                save_chunk(chunk_id, f"Local Moran's I / LISA ({indicator}, {year})", "spatial_correlation",
                           value_df, gt, modality="spatial", adjacency_path=adjacency_manifest_path,
                           insight_def_override=LOCAL_MORAN_INSIGHT_DEF, spatial_granularity="local")
                n_local += 1

    print(f"\n  -> generated {n_global} Global + {n_local} Local Moran's I chunks "
          f"(years with real values: {list(YEARS_WITH_REAL_VALUES.keys())})")


# ---------------------------------------------------------------------
# Streaming / sliding-window chunk generator (Section 4.5)
# ---------------------------------------------------------------------
# Real Cycle Hire and Air Quality data at daily/hourly grain wasn't available
# from the pasted snippets (only a few dozen rows spanning minutes/one day),
# so this demonstrates the MECHANISM on a clearly-labelled SYNTHETIC series
# with a known trend + weekly seasonality + one injected spike, using your
# proposal's exact window_days=30 / stride_days=7 parameters. Swap
# build_synthetic_streaming_series() for pd.read_csv() on your real hourly
# air-quality or daily cycle-hire data and everything else below works
# unchanged -- generate_streaming_chunks() only needs a date column and a
# value column.

def build_synthetic_streaming_series(n_days=90, seed=42):
    """SYNTHETIC demo data only -- NOT derived from any of your real
    datasets. Built purely to exercise the sliding-window mechanism:
    a rising trend + weekly (7-day) seasonality + Gaussian noise, with one
    deliberately injected spike around day 50 so at least one window has a
    genuine outlier to detect."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    trend = np.linspace(20, 35, n_days)
    weekly = 3 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    noise = rng.normal(0, 1.2, n_days)
    values = trend + weekly + noise
    values[50] += 25  # injected spike -> at least one window should flag an outlier
    return pd.DataFrame({"date": dates, "value": values})


def generate_streaming_chunks(df, date_col, value_col, dataset_name, insight_types,
                               window_days=30, stride_days=7, chunk_id_prefix="stream",
                               modality="non_spatial"):
    """Cuts df into overlapping windows (Section 4.5: 30-day windows sliding
    forward 7 days), computing ground truth independently for EACH window
    for each insight type in insight_types (subset of trend/seasonality/outlier
    -- the three that make sense to re-evaluate per rolling window). Each
    window becomes its own chunk with window_start/end/index recorded in the
    manifest so consecutive windows' verdicts can be compared afterward for
    classification-consistency analysis, exactly as Section 4.5 specifies."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    start = df[date_col].min()
    end = df[date_col].max()
    window = pd.Timedelta(days=window_days)
    stride = pd.Timedelta(days=stride_days)

    window_start = start
    idx = 0
    n_created = 0
    while window_start + window <= end + pd.Timedelta(days=1):
        window_end = window_start + window
        w = df[(df[date_col] >= window_start) & (df[date_col] < window_end)]
        if len(w) >= 4:
            for insight_type in insight_types:
                if insight_type == "trend":
                    gt = gt_trend(w[value_col].tolist())
                elif insight_type == "seasonality":
                    gt = gt_seasonality(w[value_col].tolist(), period=7)  # weekly period
                elif insight_type == "outlier":
                    gt = gt_outlier(w[date_col].astype(str).tolist(), w[value_col].tolist())
                else:
                    continue
                chunk_id = f"{chunk_id_prefix}_{insight_type}_w{idx:03d}"
                save_chunk(chunk_id, dataset_name, insight_type,
                           w[[date_col, value_col]], gt, modality=modality,
                           window_start=str(window_start.date()),
                           window_end=str(window_end.date()), window_index=idx)
                n_created += 1
        window_start += stride
        idx += 1
    print(f"\n  -> generated {n_created} streaming chunks across {idx} windows "
          f"({window_days}-day windows, {stride_days}-day stride)")


def build_streaming_demo():
    print("\n--- streaming chunks (Section 4.5 sliding-window demo) ---")
    series = build_synthetic_streaming_series()
    generate_streaming_chunks(
        series, date_col="date", value_col="value",
        dataset_name="SYNTHETIC streaming demo (not real data -- see comment above)",
        insight_types=["trend", "seasonality", "outlier"],
        window_days=30, stride_days=7,
        chunk_id_prefix="synth_stream", modality="non_spatial",
    )



if __name__ == "__main__":
    build()
