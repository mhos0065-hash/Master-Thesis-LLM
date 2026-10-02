# SpatialInsightAgent

**Evidence-Grounded Multi-Agent Reasoning for Spatiotemporal Analytics**

Cities generate spatiotemporal data faster than analysts can interpret it.
This project asks a simple question: how much architecture does an
autonomous LLM agent actually need to extract reliable insight from real
urban data?

To answer this, the project builds two things:

- **SpatialInsightBench**, a benchmark of 300+ test cases across five
  spatiotemporal insight categories, built from real London open
  government data, with every answer checked against a deterministically
  computed ground truth.
- **SpatialInsightAgent**, a three role multi-agent pipeline (a Profiler,
  a Planner, and a Coder) wired to a deterministic Validator that checks
  every answer is backed by a genuinely computed value, not a plausible
  sounding guess.

SpatialInsightAgent is compared against three baselines, a single agent
answering in one turn, the same agent given several turns to self
correct, and a ReAct agent, across eight large language models spanning
generalist and reasoning augmented families. SpatialInsightAgent achieves
the highest score in 94% of the model and metric combinations evaluated.

This repository contains the benchmark test cases, the full
SpatialInsightAgent implementation, all four evaluation configurations,
and the R and Python scripts used to compute ground truth and generate
every figure in the accompanying thesis.

---

## The Five Spatiotemporal Insight Tasks

SpatialInsightBench tests five distinct ways a model can extract meaning
from spatial data: whether a location resembles its neighbours, whether
two variables move together across space, whether a cluster of unusually
high values exists, whether one location breaks the local pattern, and
whether change over time clusters spatially. Each task below is shown
with a real worked example from the benchmark.

<table>
<tr>
<td width="50%">

**1. Spatial Correlation Analysis**
Does a location's value resemble its neighbours' values? Tested with
Local Moran's I (LISA) on 2023 recorded crime and waste generation
across London boroughs.

<img src="figures/Figure5_LISA_Crime_Waste.png" width="100%">

</td>
<td width="50%">

**2. Cross-Variable Clustering**
Do two different variables rise and fall together across neighbouring
boroughs? Tested with Lee's L on gas consumption and waste generation.

<img src="figures/Figure_CrossVariableClustering_Gas_Waste.png" width="100%">

</td>
</tr>
<tr>
<td width="50%">

**3. Spatial Hotspot Detection**
Does a location, together with its neighbours, form an unusually
intense concentration? Tested with Getis-Ord Gi* on GHG emissions.

<img src="figures/Figure_SpatialHotspotDetection_GHG.png" width="100%">

</td>
<td width="50%">

**4. Spatial Outlier Detection**
Does a location's value break sharply from what its neighbourhood
predicts? Tested with a masking-corrected regression on recorded crime.

<img src="figures/Figure_SpatialOutlierDetection_Crime.png" width="100%">

</td>
</tr>
<tr>
<td width="50%" colspan="2">

**5. Spatiotemporal Change Clustering**
Does the *change* in a variable between two time points cluster
spatially? Tested with Global Moran's I on the change in waste
generation per capita, 2021 to 2023.

<img src="figures/Figure_ChangeClusteringDetection_Waste.png" width="60%">

</td>
</tr>
</table>

Every prompt poses its question in plain, method agnostic language, a
model must choose and carry out its own statistical approach rather than
being told which method to use.

---

## Repository Structure

```
.
├── spatial analysis/          # SpatialInsightAgent implementation
│   ├── agent_utils.py         #   Profiler, Planner, Coder, Validator, execution loop
│   ├── score_*.py              #   deterministic ground truth + scoring scripts
│   └── runs/                  #   raw per-chunk execution logs
├── Follow up/summary results/ # evaluation outputs per configuration
│   ├── generate_comparison_figures_AllMethods.R   # all result figures
│   └── ...
└── figures/                   # figures used in the thesis and this README
```

## Citation

If you use SpatialInsightBench or SpatialInsightAgent, please cite the
accompanying thesis:

```
Hossain, M.T. (2026). SpatialInsightAgent: Evidence-Grounded Multi-Agent
Reasoning for Spatiotemporal Analytics. Master's thesis, Monash University.
```
