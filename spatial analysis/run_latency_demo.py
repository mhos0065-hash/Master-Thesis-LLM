"""
run_latency_demo.py
Runs multi_agent_bivariate.py once for each of two models on the SAME
chunk, times each run freshly (start/stop timestamps taken immediately
around the actual subprocess call -- never reads or reuses an earlier
measurement), and prints a short, clean, screenshot-ready summary.
On failure, the real error is printed to the console immediately (not
just saved to a log file) so a problem is never silently hidden.

Usage:
    python run_latency_demo.py
"""

import subprocess
import sys
import time

SCRIPT = "multi_agent_bivariate.py"
CHUNK = "bivariate_2021_no2_crime"
RUNS = [
    ("Reasoning (r1-0528)", "deepseek/deepseek-r1-0528", "latency_deepseek_r1.log"),
    ("Generalist (kimi-k2.6)", "moonshotai/kimi-k2.6", "latency_kimi.log"),
]

BORDER = "=" * 46


def run_one(label, model, log_path):
    print(f"  Running: {label:<24} ...", flush=True)
    cmd = [sys.executable, "-u", SCRIPT, "--model", model, "--chunk", CHUNK]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    with open(log_path, "w") as f:
        f.write(result.stdout)
        f.write(result.stderr)

    success = result.returncode == 0
    status = "done" if success else "FAILED"
    print(f"    -> {status} ({elapsed:.1f}s)")

    if not success:
        print(f"    -- error output ({log_path}) --")
        tail = (result.stderr or result.stdout).strip().splitlines()[-8:]
        for line in tail:
            print(f"    | {line}")
        print()

    return elapsed, success


def main():
    print()
    print(BORDER)
    print("  COMPUTATION COST DEMO")
    print(f"  Task: Bivariate (multi-agent)  |  Chunk: {CHUNK}")
    print(BORDER)
    print()

    results = []
    for label, model, log_path in RUNS:
        elapsed, success = run_one(label, model, log_path)
        results.append((label, elapsed, success))

    print()
    print(BORDER)
    print("  RESULT SUMMARY")
    print(BORDER)
    print()
    for label, elapsed, success in results:
        flag = "" if success else "  [FAILED]"
        print(f"  {label:<24}{elapsed:>7.1f}s  ({elapsed/60:>5.2f}m){flag}")
    print()
    print(BORDER)
    print()
    print("  Full logs: " + ", ".join(log for _, _, log in RUNS))
    print()


if __name__ == "__main__":
    main()