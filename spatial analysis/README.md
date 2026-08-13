# Spatiotemporal Insight Extraction — Experiment Harness

Everything needed to run your first end-to-end single-agent vs multi-agent
experiment, seeded with test datasets covering every QuickInsights insight
type, including sliding-window streaming chunks.

**This version fixes the methodology gaps identified against your literature
review's Section 4:** real subprocess-isolated sandbox (not in-process
`exec`), Pydantic-validated verdicts that must cite computed evidence,
Profiler caching once per dataset file, sliding-window streaming chunks,
spatial/non-spatial modality tagging, and McNemar's significance test. See
"Fixes vs. the original methodology" near the end of this file for the
full list of what changed and why.

## What's in here

```
requirements.txt         # pip install -r requirements.txt
.env.example              # copy to .env, fill in your LLM key
agent_utils.py            # chat(), agent_loop() -- delegates code execution to sandbox.py
sandbox.py                 # REAL sandbox: subprocess isolation, timeout, network blocked
schemas.py                 # Pydantic AgentVerdict -- enforces "cites computed evidence"
single_agent.py           # single-agent baseline (all 4 roles in one loop)
multi_agent.py            # Profiler -> Planner -> Coder -> Reporter, Profiler cached per dataset
run_experiment.py         # runs both modes x multiple models x all chunks
score_results.py          # precision/recall/F1 + McNemar's test vs withheld ground truth

sample_*.csv               # small hand-extracts from your 6 real datasets
generate_all_chunks.py    # builds ALL the chunks below (static + streaming) from sample_*.csv
manifest_io.py             # load_manifest()/save_manifest() -- CSV manifest helpers
chunks_csv/
    manifest.csv            # ground truth, insight defs, modality, window metadata (NOT shown to agents)
    <chunk_id>.csv          # what the Coder agent actually loads and computes on
runs/                      # created on first run: one folder per experiment
    _profiler_cache/         # one profile per unique dataset file, reused across its chunks
results.csv                # created on first run: one row per (chunk, model, mode)
scoring_summary.csv        # created by score_results.py
```

## The 48 test chunks (7 static + 14 real spatial correlation + 27 streaming)

**Static chunks — one or more per non-spatial-correlation insight type, drawn from your 6 real datasets:**

| chunk_id | dataset | insight_type | modality | ground truth |
|---|---|---|---|---|
| cyclehire_outlier_sample | Cycle Hire | outlier | non_spatial | **present** (1 journey flagged) |
| gas_outstanding_sample | Gas Consumption | outstanding_value | spatial | not present |
| crime_trend_heathrowvillages | MPS Crime | trend | non_spatial | not present |
| no2_seasonality_boysschool | NO2 Diffusion Tube | seasonality | non_spatial | not present |
| crime_evenness_heathrowvillages_202603 | MPS Crime | evenness | non_spatial | **present** (Gini=0.80) |
| leggi_attribution_cityoflondon | LEGGI | attribution | non_spatial | **present** (86.5% one sector) |
| leggi_evenness_boroughs | LEGGI | evenness | spatial | not present |

**Spatial correlation chunks — `moran_global_2023_{indicator}` and `moran_local_2023_{indicator}`,
14 chunks (7 indicators × Global + Local), all built from REAL data, not
placeholders.** These replaced an earlier version that used plain bivariate
correlation between two different variables — a genuinely different
statistical question from spatial autocorrelation, removed to avoid two
ground-truth definitions sharing one `insight_type`. Ground truth here comes
directly from your own R analysis (`spdep::moran.mc()`/`.test()` for
Global, `localmoran_perm()` for Local), transcribed from your console
output and uploaded `local_moran_results_all_years.csv`. **Independently
verified**: a from-scratch numpy reimplementation of Global Moran's I,
using only your uploaded `borough_adjacency.csv` and the real 2023 borough
values, reproduced your R-computed I to 3-4 decimal places on all 7
indicators — strong evidence the adjacency list, the transcribed values,
and the formula are all correct and mutually consistent.

Each of these 14 chunks needs **two files**, not one: `csv_path` (the
indicator's value per borough) and `adjacency_path` (the shared
`chunks_csv/borough_adjacency.csv`, borough-neighbor pairs). The Coder
agent gets both as `DATASET_PATH` and `ADJACENCY_PATH` (see
`agent_loop()`'s `extra_files` parameter in `agent_utils.py`) and must
compute Moran's I itself in pandas/numpy — no geospatial package is
available inside the sandbox (network access is blocked), so it implements
the actual weighted-sum formula directly, using permutation for
significance (mirroring your R script's Monte Carlo approach). The exact
formula and permutation recipe are in each chunk's `insight_def` in the
manifest (`GLOBAL_MORAN_INSIGHT_DEF`/`LOCAL_MORAN_INSIGHT_DEF` in
`generate_all_chunks.py`).

Only 2023 has real per-borough values right now (from the row data you
pasted in chat) — ground truth exists for 2021/2022 too (transcribed from
your R console output and the uploaded local results), but building those
years' chunks needs `PREP/2021.csv`/`PREP/2022.csv` (the raw panel files
your R script reads) uploaded so the LLM has real values to compute on. See
`YEARS_WITH_REAL_VALUES` in `generate_all_chunks.py` to add them once
available.

**Streaming chunks — `synth_stream_{trend,seasonality,outlier}_w000` through `w008`,**
27 chunks total (9 overlapping 30-day windows sliding 7 days, × 3 insight
types per window, exactly Section 4.5's parameters). Built from a
**synthetic** series with a known rising trend, weekly seasonality, and one
injected spike — clearly labelled as synthetic in `generate_all_chunks.py`,
not derived from any of your real pasted data. It exists to prove the
sliding-window *mechanism* works (verified: the injected spike is correctly
flagged as an outlier in every window that contains it, and stops being
flagged the moment it scrolls out of the window — exactly the
"classification consistency across consecutive windows" check Section 4.5
asks for). Swap `build_synthetic_streaming_series()` for `pd.read_csv()` on
your real hourly Air Quality or daily Cycle Hire data once you have it, and
everything else works unchanged.

Both positive and negative cases are included on purpose across both
groups — you need both classes to compute precision/recall/F1, per your
evaluation plan (Section 4.7).

**Important:** the `sample_*.csv` files are tiny hand-typed extracts from
the rows you pasted, for testing the *pipeline mechanics* only — not your
real datasets. One row was deliberately inserted (Cycle Hire RentalId
50754243, Duration 4980s) so the outlier chunk has a genuine positive case.
Before running your real experiment, replace `sample_*.csv` with your actual
downloaded files and re-run `generate_all_chunks.py` (or write equivalent
loaders — the ground-truth functions themselves are dataset-agnostic).

**Manifest format:** ground truth, insight definitions, modality, and
(for streaming chunks) window metadata live in `chunks_csv/manifest.csv`
(not JSON) — one row per chunk, with the insight-specific numbers (z-scores,
Gini, Pearson r, etc.) packed into a single `ground_truth_metric_json` cell
since their shape varies by insight type. Open it directly in Excel/pandas
to review or hand-edit ground truth. `manifest_io.py` provides
`load_manifest()`/`save_manifest()` so the rest of the pipeline never has to
know it's CSV under the hood.

## First-time setup — Windows PowerShell

**Fast path:** if you just want it running with minimal typing, use the
included script. Open PowerShell in the folder with all these files and run:

```powershell
powershell -ExecutionPolicy Bypass -File setup_and_run.ps1
```

This creates the virtual environment, installs dependencies, opens Notepad
for you to paste your API key into `.env`, generates the chunks if needed,
and runs the smallest possible first test automatically. Read on for the
manual steps if you'd rather understand/control each one yourself, or if the
script hits something it can't resolve.

### Manual steps

Open PowerShell in the folder where you downloaded all these files (or `cd` into it).

```powershell
# 1. Confirm Python is installed (3.10+ recommended)
python --version
# If that fails, try: py --version   (Windows sometimes only registers the 'py' launcher)

# 2. Create and activate a virtual environment (keeps this project's packages separate)
python -m venv venv
venv\Scripts\Activate.ps1
```

If step 2's activation gives an error like *"running scripts is disabled on this
system"*, PowerShell's execution policy is blocking it. Run this once (only
affects the current window, not your whole system), then retry:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1
```
Your prompt should now start with `(venv)`.

```powershell
# 3. Install dependencies
pip install -r requirements.txt

# 4. Get an API key (see the earlier discussion — Groq is free, no card:
#    https://console.groq.com/keys — or OpenRouter/OpenAI if you're using those)

# 5. Configure credentials
copy .env.example .env
notepad .env
# paste your key into LLM_API_KEY, save, close Notepad.
# agent_utils.py loads .env automatically (via python-dotenv) -- no need
# to set PowerShell environment variables by hand.

# 6. Confirm the test chunks exist (already generated for you; only needed
#    if you want to regenerate them from the sample CSVs)
python generate_all_chunks.py
```

<details>
<summary>macOS / Linux (bash/zsh) equivalent</summary>

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env          # or any editor -- paste your key into LLM_API_KEY
python3 generate_all_chunks.py
```
</details>

## Running your first experiment (PowerShell)

**Smallest possible first test — one chunk, one model, single-agent only:**
```powershell
python single_agent.py --chunk cyclehire_outlier_sample
```
Check `runs\<timestamp>_single_...\report.md` — you should see a structured
verdict (`Insight present: True/False`, a `Detail` line, and an `Evidence`
list with the actual computed number(s), not a bare FOUND/NOT FOUND). The
terminal prints the predicted verdict next to the ground truth so you can
eyeball correctness immediately. If it instead prints
`<-- NO VALID VERDICT (counts as incorrect)`, the model never produced
JSON matching the schema within the iteration limit — see troubleshooting
below.

**Same chunk through the multi-agent pipeline:**
```powershell
python multi_agent.py --chunk cyclehire_outlier_sample
```
Check `runs\<timestamp>_multi_...\` — you'll see `profiler_summary_used.txt`
(tagged `[from cache]` or `[from fresh profiler run]`), `planner.txt`,
`coder_log.json`, and `report.md`. Run it again on a **different** chunk
that happens to share the same `csv_path` (check `manifest.csv`) and you'll
see the Profiler step get skipped entirely — that's the per-dataset caching
fix in action.

**Try a streaming chunk too, to confirm those work the same way:**
```powershell
python single_agent.py --chunk synth_stream_outlier_w004
```
This is one of the sliding-window chunks that contains the injected spike —
ground truth says `insight_present: True`.

**Try a real spatial correlation (Moran's I) chunk — this is the hardest
computational task in the set, since the model must implement the Moran's I
formula itself from your adjacency list:**
```powershell
python single_agent.py --chunk moran_global_2023_crime
```
Ground truth says `insight_present: True` (real Moran's I = 0.167, p=0.006
from your R analysis). Check `runs\...\conversation_log.json` to see
whether the model's own computed I is anywhere close to 0.167 — that's a
genuinely interesting result either way (correct implementation vs. not).

**Once those look sane, run everything (this is now 48 chunks, not 8 — see
the note on cost/scope below):**
```powershell
# all 48 chunks, single-agent, default model
python single_agent.py

# all 48 chunks, multi-agent, default model
python multi_agent.py

# OR: both modes x multiple models x all chunks in one go (recommended)
python run_experiment.py --models llama-3.3-70b-versatile qwen/qwen3-32b
```

**Note on scope:** with 48 chunks × 2 models × 2 pipeline modes, a full
`run_experiment.py` sweep is 192 agent runs, each potentially several LLM
calls (code-and-reason iterations) — the 14 Moran's I chunks in particular
will likely need more iterations than the others, since implementing the
formula from scratch is a harder task. For your first real pass, consider
starting with `--chunks` set to a handful of representative chunks (one per
insight type, plus a couple of streaming windows and one Moran's I chunk)
before committing to the full sweep — especially if you're on Groq's free
tier rate limits.

**Score the results:**
```powershell
python score_results.py
```
Prints precision/recall/F1 broken down by pipeline mode, model, insight
type, AND modality (spatial vs non-spatial) — plus McNemar's test comparing
single-agent vs multi-agent correctness on matched chunks (Section 4.7).
Writes the full breakdown to `scoring_summary.csv`.

**For Local Moran's I chunks specifically, also run the borough-level scorer:**
```powershell
python score_moran_boroughs.py
```
`score_results.py`'s precision/recall/F1 only checks the top-level
FOUND/NOT FOUND verdict (was *any* significant cluster present) — it
doesn't check *which* boroughs were named. `score_moran_boroughs.py` adds
a second, stricter layer: set-based precision/recall/F1 comparing the
boroughs the model actually named against the real HH/LL boroughs from
your `local_moran_results_all_years.csv`, reported at two granularities:
- **name-only**: did it name the right borough, regardless of whether it
  also got the HH/LL/HL/LH label right for it
- **with-type**: did it name the right borough AND assign the correct
  cluster type

Both are standard set-retrieval precision/recall/F1 (the same framework
your proposal already commits to in Section 4.7), not an ad hoc scoring
rule — a model that names 1 of 2 real HH boroughs correctly scores
recall=0.5 either way, but reasoned from a citable method rather than a
hand-picked percentage. Writes `moran_borough_scoring.csv`.

## Troubleshooting the first run (PowerShell)

- `RuntimeError: LLM_API_KEY not set` — the `.env` file wasn't found or the
  key line is missing/misspelled. Confirm it loaded with:
  `python -c "import agent_utils; print(agent_utils.LLM_API_KEY)"` — if that
  prints blank, check `.env` is in the same folder you're running from (not
  a different directory), and that the line reads exactly
  `LLM_API_KEY=your_key_here` with no quotes or spaces around the `=`.
- `python : The term 'python' is not recognized...` — Python isn't on your
  PATH. Try `py` instead of `python` for every command above, or reinstall
  Python from python.org with "Add python.exe to PATH" checked.
- `venv\Scripts\Activate.ps1 cannot be loaded because running scripts is
  disabled on this system` — run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first (see
  setup step 2 above), then retry activation.
- `<-- NO VALID VERDICT (counts as incorrect)` — the model either never
  reached `FINAL ANSWER:` within `max_iterations`, or its JSON didn't match
  the schema even after one repair attempt (most commonly: an empty
  `evidence` list, since the schema requires at least one cited computed
  value). This is a real, meaningful experimental outcome (some models are
  worse at following structured-output instructions), not something to
  silently work around — but if it happens on nearly every chunk for one
  model, that model may need a higher `max_iterations` or a clearer prompt.
- `ImportError: Import of 'ftplib'/'smtplib'/'pip' is blocked` inside a run
  — this is `sandbox.py` working as intended. Only these three modules are
  blocked at import time; network access is blocked separately at the
  function level (see `sandbox.py`'s docstring for why). `os` and
  `subprocess` are deliberately NOT blocked (blocking them breaks pandas'
  internal use of them) — meaning generated code *could* in principle write
  or delete files outside `ARTIFACTS_DIR`, or spawn processes. This is a
  known, documented limitation, not a bug — see "Fixes vs. the original
  methodology" below for the honest tradeoff.
- Windows shows garbled characters (e.g. `Â°` instead of `°`) in a report —
  shouldn't happen; `sandbox.py` forces UTF-8 for the subprocess
  specifically to prevent this. If you still see it, check your PowerShell
  window's own encoding (`$OutputEncoding`), which is separate from the
  subprocess's.

## Fixes vs. the original methodology

These directly address gaps identified against your Section 4 write-up:

1. **Real sandbox, not in-process `exec`.** `sandbox.py` runs every code
   block in its own subprocess with a hard timeout and function-level
   network blocking — replacing an earlier in-process version that had no
   timeout and a write-block that pandas' own I/O could silently bypass.
   Honest residual limitation: `os`/`subprocess` aren't blocked (only
   `ftplib`/`smtplib`/`pip` are, plus network functions), so this is a
   "reasonable first line of defence," not a true security sandbox — your
   own docstring in `sandbox.py` says this outright. For a true security
   boundary, swap in a hosted sandbox service (e2b.dev has a free tier) —
   nothing else in the pipeline needs to change, since everything calls
   `run_sandboxed_code()` through one function.
2. **Verdicts must cite computed evidence.** `schemas.py`'s `AgentVerdict`
   requires a non-empty `evidence` list; a verdict with none fails Pydantic
   validation and is rejected (with one repair retry), never silently
   coerced. `predicted` in `results.csv` now comes from this validated
   field, not text-matching "FOUND"/"NOT FOUND".
3. **Pydantic-validated JSON between agents**, per Section 4.2 — implemented
   in `schemas.py`, enforced in both `single_agent.py`'s final verdict and
   `multi_agent.py`'s Reporter.
4. **Profiler runs once per dataset file**, not once per chunk (Section 4.2:
   "independent of insight type... run once per dataset"). Cached both
   in-memory and on disk (`runs/_profiler_cache/`), keyed by the dataset
   file's absolute path. Note: with today's demo chunks, each chunk_id gets
   its own CSV file (even the paired Global/Local Moran chunks for the same
   indicator, despite having identical underlying values), so the cache
   mostly won't hit yet -- the mechanism is correct and tested (confirmed:
   a second call with the same file path returns the cached profile without
   re-invoking the LLM), and will show real savings once you point several
   insight-type chunks at one shared real full dataset file, the intended
   real-experiment design.
5. **Sliding-window streaming chunks** (Section 4.5) — `generate_streaming_chunks()`
   in `generate_all_chunks.py`, using your exact 30-day window / 7-day stride
   parameters, tested and confirmed to correctly track a synthetic outlier
   scrolling through overlapping windows. Swap the synthetic demo series for
   your real Cycle Hire/Air Quality data once available.
6. **Modality tagging** (spatial vs non-spatial) added to the manifest and
   to `score_results.py`'s disaggregated breakdown, per Section 4.7.
7. **McNemar's test** added to `score_results.py`, comparing single- vs
   multi-agent correctness on matched (chunk_id, model) pairs, per
   Section 4.7.
8. **Real spatial correlation ground truth** (Global + Local Moran's I,
   14 chunks) replacing an earlier version that used plain bivariate
   correlation -- a different statistical concept from spatial
   autocorrelation. Ground truth transcribed from your actual R output and
   uploaded `local_moran_results_all_years.csv`, cross-checked against the
   printed cluster-count tables, and independently verified by reproducing
   your R-computed Global Moran's I values via a from-scratch numpy
   implementation using your uploaded adjacency list (matched to 3-4
   decimal places on all 7 indicators). See `agent_utils.py`'s
   `extra_files` parameter and `sandbox.py`'s `extra_vars` for how a second
   input file (the adjacency list) gets exposed to the Coder agent as
   `ADJACENCY_PATH` alongside the usual `DATASET_PATH`.

**Still open, and not code-fixable by me:**
- **Human ground-truth validation.** Your independent analysts should
  review a sample of the `ground_truth_*` columns in `chunks_csv\manifest.csv`
  against their own judgment, with Cohen's Kappa, before trusting them as
  the reference standard for the full run (Section 4.7). Treat the
  statistical thresholds here (z>3, Gini>0.3, r>0.3, p<0.05) as defensible
  starting points to be validated, not ground truth by fiat.
- **Real datasets for the other 6 insight types.** `sample_*.csv` (excluding
  the Moran's I files, which are now real) are still tiny hand-typed
  extracts, not your actual downloaded London Datastore/TfL files — replace
  them and re-run `generate_all_chunks.py` before any result you intend to
  report.
- **2021/2022 Moran's I chunks.** Only 2023 has real per-borough values
  right now. Ground truth for 2021/2022 is already transcribed and ready —
  upload `PREP/2021.csv`/`PREP/2022.csv` and add them to
  `YEARS_WITH_REAL_VALUES` in `generate_all_chunks.py` to unlock those.
- **`sandbox.py`'s honest residual gap** (point 1 above) — acceptable for
  LLMs that aren't adversarially trying to escape, not a genuine security
  boundary if that's a claim your methodology needs to make.
