# SpatialInsightAgent: Evidence-Grounded Multi-Agent Reasoning for Spatiotemporal Analytics

This repository accompanies a Master of IT research thesis built around
two contributions: **SpatialInsightBench**, a spatiotemporal insight
benchmark, and **SpatialInsightAgent**, a multi agent framework proposed
and evaluated on that benchmark.

## Overview

Cities generate spatiotemporal data faster than analysts can interpret
it. Large language model agents have been proposed as a way to close
this gap, but it remains unclear how much architecture such a system
actually needs. This thesis addresses that question directly.

**SpatialInsightBench** is a benchmark of more than 300 test cases
across five spatiotemporal insight categories, built from real London
open government data. Every test case has a ground truth computed
deterministically, then independently checked by a human analyst, so a
model's answer can be judged against a verifiable standard rather than a
generated reference.

**SpatialInsightAgent** is a three role multi agent framework, made up
of a Profiler, a Planner, and a Coder, coupled to a deterministic
Validator. The Validator checks that every answer is backed by a
genuinely computed value, not a fabricated or merely plausible number,
before that answer is ever accepted.

The thesis compares SpatialInsightAgent against three baselines on
SpatialInsightBench, a single agent answering in one turn, the same
agent given several turns to self correct, and a Zero-shot ReAct Baseline agent, 
across eight large language models spanning generalist and reasoning augmented
families.

## What We Found

- **SpatialInsightAgent achieves the highest score in 94 percent of the
  model and metric combinations evaluated**, across all five tasks and
  all eight models, ahead of every baseline on both accuracy and F1.
- **The gain comes from grounding, not simply better guesses.** A
  verdict only counts as correct if it is also backed by a genuinely
  computed value. Under the baselines, a meaningful share of correct
  looking answers turn out to be ungrounded, correct by chance rather
  than by computation. Under SpatialInsightAgent, this gap almost
  closes, since its genericValidator rejects attempts with no real 
  computation behind them before a verdict is produced.
- **Letting a single agent self correct over several turns barely
  helps.** Without an external, verifiable check on its own output, a
  model that repeats its own reasoning shows only a modest gain over
  answering in one pass, consistent with prior findings that self
  correction needs real, outside feedback to work.
- **Reasoning augmented models generally outperform generalist models,
  but not always.** The gap is widest under SpatialInsightAgent and
  ReAct, where the agent receives real feedback to act on. Under the
  simpler baselines, the strongest generalist model beats every
  reasoning augmented model tested, so model family alone does not
  decide performance, how the agent is configured matters just as much.
- **Task difficulty varies sharply.** Models perform best on spatial
  hotspot and spatial outlier detection, and consistently worst on
  cross variable clustering, across every configuration tested.

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
Does a location's value resemble its neighbours' values? The attached figure is tested with
Local Moran's I (LISA) on 2023 recorded crime and waste generation
across London boroughs.

<img src="Follow up/summary results/Appendix/figures/Figure5_LISA_Crime_Waste.png" width="100%">

</td>
<td width="50%">

**2. Cross Variable Clustering**
Do two different variables rise and fall together across neighbouring
boroughs? The attached figure is tested with Lee's L on gas consumption and waste generation.

<img src="Follow up/summary results/Appendix/figures/Figure_CrossVariableClustering_Gas_Waste.png" width="100%">

</td>
</tr>
<tr>
<td width="50%">

**3. Spatial Hotspot Detection**
Does a location, together with its neighbours, form an unusually
intense concentration? The figure here is tested with Getis Ord Gi* on GHG emissions.

<img src="Follow up/summary results/Appendix/figures/Figure_SpatialHotspotDetection_GHG.png" width="100%">

</td>
<td width="50%">

**4. Spatial Outlier Detection**
Does a location's value break sharply from what its neighbourhood
predicts? The figure here is tested with a masking corrected regression on recorded crime.

<img src="Follow up/summary results/Appendix/figures/Figure_SpatialOutlierDetection_Crime.png" width="100%">

</td>
</tr>
<tr>
<td width="50%" colspan="2">

**5. Spatiotemporal Change Clustering**
Does the *change* in a variable between two time points cluster
spatially? The attached figure is tested with Global Moran's I on the change in waste
generation per capita, 2021 to 2023.

<img src="Follow up/summary results/Appendix/figures/Figure_ChangeClusteringDetection_Waste.png" width="60%">

</td>
</tr>
</table>

Every prompt poses its question in plain, method agnostic language, so a
model must choose and carry out its own statistical approach rather than
being told which method to use.

---

## Repository Structure

```
.
├── spatial analysis/          # SpatialInsightAgent implementation
│   ├── agent_utils.py         #   Profiler, Planner, Coder, Validator, execution loop
│   ├── score_*.py              #   deterministic ground truth and scoring scripts
│   └── runs/                  #   raw per chunk execution logs
├── Follow up/summary results/ # evaluation outputs per configuration
│   ├── generate_comparison_figures_AllMethods.R   # all result figures
│   └── Appendix/figures/      #   worked example figures for each task
└── README.md
```

## Citation

If you use SpatialInsightBench or SpatialInsightAgent, please cite the
accompanying thesis:

```
Hossain, M.T. and Ali, M.E. (2026). SpatialInsightAgent: Evidence
Grounded Multi Agent Reasoning for Spatiotemporal Analytics. Master's
thesis, Monash University.
```
