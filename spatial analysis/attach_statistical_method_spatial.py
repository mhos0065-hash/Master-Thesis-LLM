"""
attach_statistical_method_spatial.py
======================================
Same approach as the other tasks' attach scripts. Reads the 3 spatial-
correlation results CSVs (Global + Local Moran's I in one dataset,
distinguished by the `spatial_granularity` column), locates each row's
run folder on disk, scans every log/code file inside it for evidence of
which method was actually used, and writes the result into a new
`statistical_method` column. Writes "none" if nothing is found.

Detection patterns are shared across both granularities -- the row's
OWN `spatial_granularity` column (already in the results CSV) is what
determines which half of the ground-truth table a detected method gets
looked up against, not the detection logic itself.

USAGE
-----
1. Edit RUNS_DIR and RESULT_FILES below.
2. Run: python attach_statistical_method_spatial.py
3. Output: "<name>_with_method.csv" next to each input file, plus
   "unmatched_rows_spatial.csv" for any row whose run folder wasn't found.
"""

import csv
import re
from pathlib import Path

# ============================= CONFIG ============================
RUNS_DIR = Path(r"D:\Australia\Monash\Semester 4\Research stream\Master Thesis\Master-Thesis-LLM\spatial analysis\runs")

RESULT_FILES = {
    "single":     Path(r"D:\Australia\Monash\Semester 4\Research stream\Master Thesis\Master-Thesis-LLM\spatial analysis\spatial_results.csv"),
    "multi":      Path(r"D:\Australia\Monash\Semester 4\Research stream\Master Thesis\Master-Thesis-LLM\spatial analysis\results\single agent\results_spatial_multi.csv"),
    "multiagent": Path(r"D:\Australia\Monash\Semester 4\Research stream\Master Thesis\Master-Thesis-LLM\spatial analysis\results\multi agent\spatial_correlation_results_multiagent.csv"),
}

LOG_EXTENSIONS = {".txt", ".log", ".json", ".md", ".py", ".out", ".jsonl"}
MAX_BYTES_PER_FILE = 5_000_000
# ===================================================================

# Ordered (label, regex). "Local Moran's I / LISA" checked BEFORE the
# generic "Global Moran's I" pattern for the same reason as
# changespillover: a bare "moran'?s?\s*i" regex would otherwise also
# match inside "local moran's i" text.
METHOD_PATTERNS = [
    ("Local Moran's I / LISA",       r"local\s+moran|\blisa\b"),
    ("Getis-Ord Gi / Gi*",           r"getis[-\s_]?ord|gi\*|g_i\*"),
    ("Geary's C",                    r"geary'?s?\s*c\b|geary_c\b"),
    ("Join Count Statistic",         r"join[\s_-]?count"),
    ("Neighbor-average correlation (non-standard)",
     r"neighbor.{0,20}(average|avg).{0,20}(correl|pearson)|correl.{0,20}neighbor"),
    ("Global Moran's I",             r"moran'?s?\s*i\b|moran_i\b|morans_i\b"),
    ("Permutation / Monte Carlo test (unspecified statistic)",
     r"permutation|monte\s+carlo|randomiz"),
]
METHOD_REGEXES = [(label, re.compile(pat, re.IGNORECASE)) for label, pat in METHOD_PATTERNS]

_TIMESTAMP_RE = re.compile(r"(\d{8}T\d{6}Z)_")


def find_run_folder(run_dir_value: str):
    if not run_dir_value:
        return None
    parts = re.split(r"[\\/]+", run_dir_value.strip())
    folder_name = parts[-1]

    candidate = RUNS_DIR / folder_name
    if candidate.is_dir():
        return candidate

    m = _TIMESTAMP_RE.match(folder_name)
    if m and RUNS_DIR.is_dir():
        timestamp = m.group(1)
        for entry in RUNS_DIR.iterdir():
            if entry.is_dir() and entry.name.startswith(timestamp):
                return entry
    return None


def scan_folder_for_method(folder: Path) -> str:
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
                "spatial_granularity": row.get("spatial_granularity", ""),
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
        print(f"ERROR: RUNS_DIR does not exist: {RUNS_DIR}")
        return

    unmatched_rows = []
    for label, path in RESULT_FILES.items():
        if not path.exists():
            print(f"WARNING: {label} file not found at {path}, skipping.")
            continue
        process_file(label, path, unmatched_rows)

    if unmatched_rows:
        out = Path("unmatched_rows_spatial.csv")
        with open(out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source_file", "chunk_id", "model", "mode",
                                                     "spatial_granularity", "run_dir"])
            writer.writeheader()
            writer.writerows(unmatched_rows)
        print(f"\n{len(unmatched_rows)} rows had no matching run folder -- see {out.resolve()}")
    else:
        print("\nAll rows matched a run folder on disk.")


if __name__ == "__main__":
    main()
