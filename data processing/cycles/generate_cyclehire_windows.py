"""
generate_cyclehire_windows.py
================================
Splits the real Cycle Hire journey dataset (journeys_with_boroughs.csv)
into overlapping 7-day sliding windows (1-day stride -- chosen to maximize
usable windows from the available 13-day real date range: Jan 10-23, 2016,
giving 8 overlapping windows). Each window is saved as its own CSV,
containing only the columns needed for the self-containment analysis
(Rental Id, Start Date, StartBorough, EndBorough).

Also writes windows_index.csv, listing every window's id/date range/trip
count/filepath -- the R script reads this to know what to iterate over.

Usage:
    python generate_cyclehire_windows.py
Output:
    cyclehire_windows/window_w000.csv ... window_w007.csv
    cyclehire_windows/windows_index.csv
"""

import pandas as pd
import os

INPUT_CSV = "/mnt/user-data/uploads/journeys_with_boroughs.csv"
OUT_DIR = "cyclehire_windows"
WINDOW_DAYS = 7
STRIDE_DAYS = 1

os.makedirs(OUT_DIR, exist_ok=True)


def run():
    df = pd.read_csv(INPUT_CSV, usecols=["Rental Id", "Start Date", "StartBorough", "EndBorough"])
    df["Start Date"] = pd.to_datetime(df["Start Date"], format="%d/%m/%Y %H:%M")
    # drop rows with missing borough (station coordinates outside the
    # spatial join area, or missing lat/lon in the source) -- can't assign
    # these to any borough for the spatial analysis
    n_before = len(df)
    df = df.dropna(subset=["StartBorough", "EndBorough"])
    print(f"Dropped {n_before - len(df)} rows with missing StartBorough/EndBorough "
          f"({len(df)} rows remain)")

    start = df["Start Date"].min().normalize()
    end = df["Start Date"].max()
    window = pd.Timedelta(days=WINDOW_DAYS)
    stride = pd.Timedelta(days=STRIDE_DAYS)

    index_rows = []
    window_start = start
    idx = 0
    while window_start + window <= end + pd.Timedelta(days=1):
        window_end = window_start + window
        w = df[(df["Start Date"] >= window_start) & (df["Start Date"] < window_end)]

        filename = f"window_w{idx:03d}.csv"
        filepath = os.path.join(OUT_DIR, filename)
        w.to_csv(filepath, index=False)

        index_rows.append({
            "window_id": f"w{idx:03d}",
            "window_start": window_start.strftime("%Y-%m-%d"),
            "window_end": (window_end - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),  # inclusive end date
            "n_trips": len(w),
            "filepath": filename,
        })
        print(f"window_w{idx:03d}: {window_start.date()} to {(window_end - pd.Timedelta(days=1)).date()} "
              f"-> {len(w)} trips -> {filepath}")

        window_start += stride
        idx += 1

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(os.path.join(OUT_DIR, "windows_index.csv"), index=False)
    print(f"\n{idx} windows written to {OUT_DIR}/")
    print(f"Index written to {OUT_DIR}/windows_index.csv")


if __name__ == "__main__":
    run()
