# Spatiotemporal Insight Extraction with LLM Agents for Urban Sustainability: The Role of Multi-Agent Pipelines

This repository accompanies the paper *SpatialInsightBench: Benchmarking LLM Agents for
Spatiotemporal Insight Extraction from Urban Open Data*. It has two
contributions: **SpatialInsightBench**, a spatiotemporal insight
benchmark, and **SpatialInsightAgent**, a multi agent framework
proposed and evaluated on that benchmark.

## Overview

Cities generate spatiotemporal data faster than analysts can interpret
it. Large language model agents have been proposed as a way to close
this gap, but it remains unclear how much architecture such a system
actually needs. This work addresses that question directly.

**SpatialInsightBench** is a benchmark of more than 300 test cases
across five spatial and spatiotemporal insight categories, built from
real London open government data. The application domain is urban
sustainability: the six datasets cover air quality, energy, emissions,
waste, crime and cycle hire. Every test case has a ground truth computed
deterministically, then independently checked by a human analyst, so a
model's answer can be judged against a verifiable standard rather than a
generated reference.

**SpatialInsightAgent** is a three role multi agent framework, made up
of a Profiler, a Planner, and a Coder, coupled to a deterministic
Validator. The Validator applies five generic checks to every attempt,
for example that the reported statistic is a finite computed value and
that the code reproduces it when re-run. It never checks the correct
answer or method.

The study compares SpatialInsightAgent against three baselines on
SpatialInsightBench, a single agent answering in one turn, the same
agent given several turns to self correct, and a zero-shot ReAct baseline, 
across eight large language models spanning generalist and reasoning augmented
families.

## What We Found

- **SpatialInsightAgent achieves the highest score in 94 percent of the
  model, task and metric combinations evaluated**, and the highest mean
  F1 for all eight models.
- **The gain comes from grounding, not simply better guesses.** A
  verdict only counts as correct if it is also backed by a genuinely
  computed value. Under the baselines, a meaningful share of correct
  looking answers turn out to be ungrounded, correct by chance rather
  than by computation. Under SpatialInsightAgent, far fewer
  correct answers are ungrounded, consistent with its Validator
  rejecting attempts with no real computation behind them. It does have
  the most reasoning failures, because the Validator checks that a value
  was computed, not that the method was right.
- **Letting a single agent self correct over several turns barely
  helps.** Without an external, verifiable check on its own output, a
  model that repeats its own reasoning shows only a modest gain over
  answering in one pass, consistent with prior findings that self
  correction needs real, outside feedback to work.
- **Reasoning augmented models are ahead of generalist models in most
  comparisons, but not all.** The gap is widest under
  SpatialInsightAgent. Under the two simplest baselines, the strongest
  generalist model, Kimi-K2.6, beats every reasoning augmented model, so
  model family alone does not decide performance.
- **Task difficulty varies sharply.** Spatial correlation and cross
  variable clustering are the two hardest categories in every
  configuration. Spatial outlier detection is the easiest under the
  three baselines, and spatiotemporal change clustering under
  SpatialInsightAgent.

---

## The Five Insight Categories

SpatialInsightBench tests five distinct ways a model can extract meaning
from spatial data: whether a location resembles its neighbours, whether
two variables move together across space, whether a cluster of unusually
high values exists, whether one location breaks the local pattern, and
whether change over time clusters spatially. Each task below is shown
with a real worked example from the benchmark.

<table>
<tr>
<td width="50%" valign="top">

**1. Spatial Correlation Analysis**
Does a location's value resemble its neighbours' values? Shown here:
Local Moran's I (LISA) on 2023 recorded crime and waste generation
across London boroughs.

<img src="Evaluations/Experimental Results/Appendix/figures/Figure5_LISA_Crime_Waste.png" width="100%">

</td>
<td width="50%" valign="top">

**2. Cross Variable Clustering**
Do two different variables rise and fall together across neighbouring
boroughs? Shown here: Lee's L on gas consumption and waste generation.

<img src="Evaluations/Experimental Results/Appendix/figures/Figure_CrossVariableClustering_Gas_Waste_2.png" width="100%">

</td>
</tr>
<tr>
<td width="50%" valign="top">

**3. Spatial Hotspot Detection**
Does a location, together with its neighbours, form an unusually
intense concentration? Shown here: Getis Ord Gi* on GHG emissions.

<img src="Evaluations/Experimental Results/Appendix/figures/Figure_SpatialHotspotDetection_GHG.png" width="100%">

</td>
<td width="50%" valign="top">

**4. Spatial Outlier Detection**
Does a location's value break sharply from what its neighbourhood
predicts? Shown here: a masking corrected regression on recorded crime.

<img src="Evaluations/Experimental Results/Appendix/figures/Figure_SpatialOutlierDetection_Crime.png" width="100%">

</td>
</tr>
<tr>
<td colspan="2" valign="top">

**5. Spatiotemporal Change Clustering**
Does the *change* in a variable between two time points cluster
spatially? Shown here: Global Moran's I on the change in waste
generation per capita, 2021 to 2023.

<p align="center">
  <img src="Evaluations/Experimental Results/Appendix/figures/Figure_ChangeClusteringDetection_Waste.png" width="80%">
</p>

</td>
</tr>
</table>

Every prompt poses its question in plain language and does not force a
statistical method, so a model may choose any suitable one.

---

## Repository Structure

```
.
├── spatial analysis/                 # SpatialInsightAgent implementation and experiments
│   ├── code/                         #   all source code
│   │   ├── test case generations/    #     builds the SpatialInsightBench test cases
│   │   ├── ground truth/             #     deterministic ground truth computation
│   │   ├── single and multi turn/    #     Configurations I and II (single agent)
│   │   ├── ReAct/                    #     Configuration III (zero-shot ReAct baseline)
│   │   ├── Multi agent/              #     Configuration IV (SpatialInsightAgent)
│   │   ├── score/                    #     evidence-grounded scoring scripts
│   │   ├── Error analysis/           #     error taxonomy classification
│   │   └── debug/                    #     debugging utilities
│   ├── data pre-processing/          #   cleaning and normalisation of the six datasets
│   ├── chunks_csv/                   #   per test case data chunks files given to the models
│   ├── ground truth/                 #   computed ground truth labels
│   ├── runs_logs/                    #   raw execution run logs (single, multi turn and SpatialInsightAgent)
│   ├── ReAct_runs/                   #   raw execution logs (ReAct baseline)
│   ├── logs_static/                  #   static run logs
│   ├── results/                      #   model outputs per configuration
│   ├── Scores/                       #   Evaluation scores for all configurations
│   ├── Errors/                       #   error taxonomy outputs including quantitative error analysis
│   ├── Qualitative Error Analysis/   #   case study logs and transcripts
│   ├── artifacts/                    #   intermediate files produced during runs
│   ├── .env.example                  #   template for API keys
│   └── setup_and_run.ps1             #   environment setup and run script (Windows)
├── Evaluations/                      # evaluation outputs and figures
│   └── Experimental Results/         #   result tables and comparison figures
│       ├── code/                     #     R code for experimental result analysis
│       └── Appendix/                 #     R code, CSV files and figures for the Appendix
├── .gitignore
└── README.md
```

