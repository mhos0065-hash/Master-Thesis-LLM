"""
attach_statistical_method_getisord.py
=======================================
Same approach as attach_statistical_method.py (used for the bivariate
task), adapted for Getis-Ord hotspot detection. Reads the 3 getisord
results CSVs, locates each row's run folder on disk, scans every
log/code file inside it for evidence of which hotspot-detection method
was actually used, and writes the result into a new `statistical_method`
column. Writes "none" if nothing is found or the folder can't be located.

IMPORTANT DIFFERENCE FROM BIVARIATE: evidence_text in this task often
already contains a label ("z-score = ..." or "Gi*= ..."), but that label
is frequently just the REQUIRED OUTPUT FORMAT'S example placeholder
("z-score" is the example given in the task prompt itself), not a
reliable indicator of what was actually computed -- e.g. a naive,
non-standard combined-neighbor-sum calculation (a real bug seen in this
project's transcripts) also prints "z-score = ...". So this script
still scans the full run folder's code/logs, not just evidence_text.

USAGE
-----
1. Edit RUNS_DIR and RESULT_FILES below.
2. Run: python attach_statistical_method_getisord.py
3. Output: "<name>_with_method.csv" next to each input file, plus
   "unmatched_rows_getisord.csv" listing any row whose run folder
   couldn't be found on disk.
"""

import csv
import re
from pathlib import Path

# ============================= CONFIG ============================
RUNS_DIR = Path(r"D:\Australia\Monash\Semester 4\Research stream\Master Thesis\Master-Thesis-LLM\spatial analysis\runs")

RESULT_FILES = {
    "ReAct":      Path(r"D:\Australia\Monash\Semester 4\Research stream\Master Thesis\Master-Thesis-LLM\spatial analysis\ReAct\getisord_results_react.csv"),
    "single":     Path(r"D:\Australia\Monash\Semester 4\Research stream\Master Thesis\Master-Thesis-LLM\spatial analysis\getisord_results.csv"),
    "multi":      Path(r"D:\Australia\Monash\Semester 4\Research stream\Master Thesis\Master-Thesis-LLM\spatial analysis\results\single agent\getisord_results_multi.csv"),
    "multiagent": Path(r"D:\Australia\Monash\Semester 4\Research stream\Master Thesis\Master-Thesis-LLM\spatial analysis\results\multi agent\getisord_results_multiagent.csv"),
}

LOG_EXTENSIONS = {".txt", ".log", ".json", ".md", ".py", ".out", ".jsonl"}
MAX_BYTES_PER_FILE = 5_000_000
# ===================================================================

# Ordered (label, regex). More specific patterns first -- a run's code
# may mention several things in passing (e.g. computing a global mean
# as a side calculation), so the label should reflect the method that
# actually produced the FINAL flagged boroughs, which is best captured
# by checking the more distinctive/rare terms before generic ones.
#
# Getis-Ord star-detection is a 2-step check: first look for an
# explicit "*" near "G"/"Gi" (Gi*, G*, gi_star), THEN fall back to a
# bare "Getis-Ord" mention with no star evidence (ambiguous which
# variant -- likely Gi, since spdep's default nb2listw has no self-loop
# unless include.self() is explicitly called).
METHOD_PATTERNS = [
    ("Getis-Ord Gi*",              r"gi\s*\*|g\s*\*\s*statistic|gi_star|g_i\^?\*"),
    ("Getis-Ord Gi (no star)",     r"getis[-\s_]?ord"),
    ("Local Moran's I / LISA",     r"local\s+moran|\blisa\b"),
    ("IQR / percentile outlier",   r"\biqr\b|interquartile|percentile"),
    ("DBSCAN / clustering",        r"dbscan|k-?means|hierarchical\s+clustering"),
    ("Kernel density estimation",  r"kernel\s+density|\bkde\b"),
    ("Standard deviational ellipse", r"deviational\s+ellipse"),
    # The exact bug pattern seen in this project's own transcripts:
    # summing a borough + its neighbors' raw values and z-scoring that
    # sum against the CITYWIDE mean/std of individual values -- not a
    # real statistic, just an ad hoc calculation that happens to also
    # print "z-score = ...".
    ("Combined-sum / naive neighbor-sum (non-standard)",
     r"combined[_\s]?(value|crime|sum)|neighbor.{0,15}sum|sum.{0,15}neighbor"),
    # Catch-all: SOME z-score was computed, but none of the more
    # specific patterns above matched -- genuinely unclear which method.
    ("Generic z-score (unspecified method)", r"z[-_]?score|zscore"),
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
        out = Path("unmatched_rows_getisord.csv")
        with open(out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source_file", "chunk_id", "model", "mode", "run_dir"])
            writer.writeheader()
            writer.writerows(unmatched_rows)
        print(f"\n{len(unmatched_rows)} rows had no matching run folder -- see {out.resolve()}")
    else:
        print("\nAll rows matched a run folder on disk.")


if __name__ == "__main__":
    main()
