"""
attach_statistical_method.py
=============================
Reads your 3 results CSVs (single / multi-turn / multiagent), locates
each row's run folder on disk (using the `run_dir` column), scans every
log/code file inside that ONE folder for evidence of which statistical
method was actually used, and writes the result into a new
`statistical_method` column. Writes "none" if nothing is found or the
folder can't be located.

This only opens the small set of files inside each row's own run
folder -- never the whole runs/ directory at once -- so it stays fast
even if you have thousands of run folders on disk.

USAGE
-----
1. Edit the CONFIG block below: set RUNS_DIR to the folder that
   directly contains the run subfolders (the "runs" folder itself, the
   one from your screenshot), and set the 3 CSV paths.
2. Run:  python attach_statistical_method.py
3. Output: for each input CSV, a new file named
   "<original_name>_with_method.csv" is written next to it, with an
   added `statistical_method` column. Originals are never modified.

A summary is printed at the end, and any rows whose run folder could
NOT be located on disk are written to "unmatched_rows.csv" so you can
inspect / fix path issues.
"""

import csv
import re
from pathlib import Path

# ============================= CONFIG =============================

# The folder that directly contains run subfolders, e.g.:
#   .../Master-Thesis-LLM/spatial analysis/runs
RUNS_DIR = Path(r"D:\Australia\Monash\Semester 4\Research stream\Master Thesis\Master-Thesis-LLM\spatial analysis\runs")

# Your 3 results CSVs (edit paths to match your machine)
RESULT_FILES = {
    "single":     Path(r"D:\Australia\Monash\Semester 4\Research stream\Master Thesis\Master-Thesis-LLM\spatial analysis\bivariate_results.csv"),
    "multi":      Path(r"D:\Australia\Monash\Semester 4\Research stream\Master Thesis\Master-Thesis-LLM\spatial analysis\results\single agent\bivariate_results_multi.csv"),
    "multiagent": Path(r"D:\Australia\Monash\Semester 4\Research stream\Master Thesis\Master-Thesis-LLM\spatial analysis\results\multi agent\bivariate_results_multiagent.csv"),
}

# Which file extensions inside a run folder count as "log/code" and get
# searched. Add more if your logs use a different extension.
LOG_EXTENSIONS = {".txt", ".log", ".json", ".md", ".py", ".out", ".jsonl"}

# Safety cap so a single huge file doesn't blow up memory (per file, in bytes)
MAX_BYTES_PER_FILE = 5_000_000

# ====================================================================

# Ordered list of (label, regex). Order matters: more specific /
# bivariate-relevant patterns are checked first, since a run's code
# may also compute univariate stats as an intermediate step and we
# want the label that best reflects the actual test used for the
# verdict, not an incidental helper calculation.
METHOD_PATTERNS = [
    ("Lee's L",                          r"lee'?s?\s*l\b|lee_l\b|lee\.l\b|leeL\b"),
    ("Bivariate / Cross Moran's I",       r"bivariate\s+moran|cross[\s_-]?moran"),
    ("Moran's I",                        r"moran'?s?\s*i\b|moran_i\b|morans_i\b"),
    ("Geary's C",                        r"geary'?s?\s*c\b|geary_c\b"),
    ("Join Count Statistic",             r"join[\s_-]?count"),
    ("LISA / Local Indicator",           r"\blisa\b|local\s+indicator"),
    ("Getis-Ord (G / G*)",               r"getis[-\s_]?ord|\bg\*?\s*statistic\b"),
    ("Mantel Test",                      r"mantel"),
    ("Spearman correlation",             r"spearman"),
    ("Kendall correlation",              r"kendall"),
    ("Pearson correlation",              r"pearson"),
    ("Chi-square test",                  r"chi.?square|chi2_contingency|chisquare"),
    ("t-test",                           r"\bt-test\b|ttest_ind|ttest_rel"),
    ("Regression-based test",            r"\bols\b|linregress|regression\s+coefficient"),
    ("Permutation / Monte Carlo test",   r"permutation\s+test|monte\s+carlo|randomi[sz]ation\s+test"),
    ("z-score / z-test",                 r"z-score|z_score|ztest|z-test"),
    ("Generic Pearson (np.corrcoef)",    r"np\.corrcoef|numpy\.corrcoef|corrcoef\("),
]
METHOD_REGEXES = [(label, re.compile(pat, re.IGNORECASE)) for label, pat in METHOD_PATTERNS]

_TIMESTAMP_RE = re.compile(r"(\d{8}T\d{6}Z)_")


def find_run_folder(run_dir_value: str):
    """Locate the on-disk folder for a given `run_dir` CSV value."""
    if not run_dir_value:
        return None

    # Normalise Windows/posix separators, take the folder name itself
    parts = re.split(r"[\\/]+", run_dir_value.strip())
    folder_name = parts[-1]

    # 1) Exact name match
    candidate = RUNS_DIR / folder_name
    if candidate.is_dir():
        return candidate

    # 2) Fallback: match by timestamp prefix alone (always unique),
    #    in case the on-disk tag differs from what's in the CSV
    #    (e.g. CSV says "bivariate_..." but disk says "spatialcorr_...")
    m = _TIMESTAMP_RE.match(folder_name)
    if m and RUNS_DIR.is_dir():
        timestamp = m.group(1)
        for entry in RUNS_DIR.iterdir():
            if entry.is_dir() and entry.name.startswith(timestamp):
                return entry

    return None


def scan_folder_for_method(folder: Path) -> str:
    """Search every text/code/log file in `folder` (recursively) for
    the first matching statistical-method keyword, in priority order."""
    if folder is None or not folder.is_dir():
        return "none"

    combined_chunks = []
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in LOG_EXTENSIONS:
            try:
                data = path.read_bytes()[:MAX_BYTES_PER_FILE]
                combined_chunks.append(data.decode("utf-8", errors="ignore"))
            except Exception:
                continue

    if not combined_chunks:
        return "none"

    combined = "\n".join(combined_chunks)
    for label, regex in METHOD_REGEXES:
        if regex.search(combined):
            return label

    return "none"


def process_file(label: str, csv_path: Path, unmatched_rows: list):
    print(f"\nProcessing [{label}]: {csv_path.name}")
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        if "statistical_method" not in fieldnames:
            fieldnames.append("statistical_method")
        rows = list(reader)

    n_found = 0
    n_no_folder = 0
    for i, row in enumerate(rows, 1):
        folder = find_run_folder(row.get("run_dir", ""))
        if folder is None:
            n_no_folder += 1
            unmatched_rows.append({
                "source_file": csv_path.name,
                "chunk_id": row.get("chunk_id", ""),
                "model": row.get("model", ""),
                "mode": row.get("mode", ""),
                "run_dir": row.get("run_dir", ""),
            })
        method = scan_folder_for_method(folder)
        row["statistical_method"] = method
        if method != "none":
            n_found += 1
        if i % 50 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} rows processed...")

    out_path = csv_path.with_name(csv_path.stem + "_with_method" + csv_path.suffix)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  -> method found for {n_found}/{len(rows)} rows")
    print(f"  -> run folder NOT found for {n_no_folder}/{len(rows)} rows")
    print(f"  -> saved: {out_path}")


def main():
    if not RUNS_DIR.is_dir():
        print(f"ERROR: RUNS_DIR does not exist or is not a folder:\n  {RUNS_DIR}")
        print("Edit the RUNS_DIR path at the top of this script and re-run.")
        return

    unmatched_rows = []
    any_processed = False
    for label, path in RESULT_FILES.items():
        if not path.exists():
            print(f"WARNING: {label} file not found at {path}, skipping.")
            continue
        process_file(label, path, unmatched_rows)
        any_processed = True

    if not any_processed:
        print("\nNo CSV files were processed. Check RESULT_FILES paths at the top of this script.")
        return

    if unmatched_rows:
        out = Path("unmatched_rows.csv")
        with open(out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source_file", "chunk_id", "model", "mode", "run_dir"])
            writer.writeheader()
            writer.writerows(unmatched_rows)
        print(f"\n{len(unmatched_rows)} rows had no matching run folder on disk -- see {out.resolve()}")
    else:
        print("\nAll rows matched a run folder on disk.")


if __name__ == "__main__":
    main()
