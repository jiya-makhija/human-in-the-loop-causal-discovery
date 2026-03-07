# human-in-the-loop-causal-discovery

Replication and ablation study of **LLM-guided causal graph discovery**.  
This project investigates how structured LLM interventions — **edge pruning, direction correction, and missing-edge discovery** — improve the accuracy of causal graph recovery.

The system combines a **BFS-style causal discovery algorithm** with **Gemini LLM reasoning** to iteratively construct and refine a directed acyclic causal graph.

---

## Abstract

This project studies how **large language models (LLMs)** can assist causal discovery algorithms by acting as structured reasoning modules.  

We compare a traditional **algorithmic causal discovery baseline (PC algorithm)** with a pipeline that introduces three LLM-guided interventions:

- edge pruning
- direction verification
- missing-edge discovery

The goal is to measure how these interventions affect **graph recovery accuracy** using standard causal discovery metrics.

Experiments are performed on the **Asia Bayesian Network**, a well-known benchmark with known ground-truth causal structure.

---

# Method Overview

The causal discovery pipeline consists of four stages:

1. **Baseline Graph Construction**
   - Identify root variables using the LLM
   - Expand the graph using BFS exploration
   - Propose candidate causal edges
   - Verify direct causality using LLM checks

2. **Edge Pruning**
   - Remove edges that represent indirect relationships.

3. **Direction Correction**
   - Verify whether discovered edges are oriented correctly.

4. **Missing Edge Discovery**
   - Ask the LLM to propose important causal relationships not discovered during BFS.

Each stage incrementally improves the accuracy of the recovered causal graph.

---

# Project Structure

This project uses both **algorithmic causal discovery** and **LLM-based intervention**.

## `src/run_experiment.py`

Entry point for running experiments.

Responsibilities:

- Load the dataset
- Run the selected baseline
- Apply one or more intervention steps
- Compute evaluation metrics
- Save experiment results to the `results/` folder

Run with:

```bash
python -m src.run_experiment
```

---

## `src/baseline_bfs.py`

Implements the original **LLM-guided BFS baseline**.

Steps:

1. Ask the LLM for likely root variables.
2. For each variable `X`, ask which variables `X` directly causes.
3. Prevent cycles while constructing the graph.
4. Build a predicted directed acyclic graph (DAG).

This file represents the original LLM-first graph construction approach.

---

## `src/causal_baseline.py`

Implements the **algorithmic causal discovery baseline**.

Currently uses the **PC algorithm** to recover a causal graph from sampled Asia network data.

This provides a non-LLM baseline that can then be refined using LLM interventions.

---

## `src/intervention.py`

Contains the intervention steps used to refine the graph after baseline discovery.

Main functions:

- `prune_indirect_edges(...)` → removes indirect or implausible causal edges
- `correct_edge_directions(...)` → keeps, flips, or removes incorrectly oriented edges
- `suggest_missing_edges(...)` → proposes additional missing direct causal edges

These functions implement the main contribution of the project.

---

## `src/llm_interface.py`

Interface for interacting with the Gemini API.

Responsibilities:

- Prompt construction
- JSON-only response parsing
- Retry logic (including rate-limit handling)
- API usage tracking
- Response caching

Key functions:

- `get_root_nodes()` → proposes likely root causes
- `get_children(node)` → predicts variables caused by a node
- `verify_edge_direct(src, dst)` → checks whether an edge is a direct causal relationship
- `verify_edge_direction(src, dst)` → determines if an edge should be flipped or removed
- `suggest_missing_edges(current_edges, nodes)` → proposes missing causal relationships

---

## `src/datasets.py`

Loads datasets used in the experiments.

Includes:

- a small toy dataset for debugging
- the **Asia Bayesian Network** structure
- generated Asia samples for running the PC baseline

This file makes it possible to compare graph recovery against known ground truth.

---

## `src/graph.py`

Simple directed graph implementation.

Handles:

- storing nodes and edges
- checking for duplicate edges
- cycle detection

Ensures the discovered graph remains a valid **DAG**.

---

## `src/cache.py`

Disk caching system for LLM responses.

Stores responses in `.cache_gemini` so repeated runs reuse previous answers instead of calling the API again.

Benefits:

- faster experiments
- reduced API cost
- avoids rate limits

---

## `src/logger.py`

Tracks experiment statistics and debugging information.

Logs:

- edges added
- edges rejected
- cycle prevention
- duplicate edges

Also stores counts used for final experiment summaries.

---

## `src/metrics.py`

Computes evaluation metrics for graph recovery.

Includes:

- precision
- recall
- F1 score
- structural hamming distance (SHD)

These metrics are used to compare the baseline and intervention pipelines.

---

## `results/`

Stores experiment outputs.

Each run saves a JSON file containing:

- predicted edges
- ground truth edges
- precision, recall, and F1 score
- structural hamming distance (SHD)
- LLM usage statistics

---

## `.cache_gemini/`

Local cache for Gemini responses.

This folder stores cached LLM outputs so repeated runs reuse previous answers without calling the API again.

This significantly speeds up experimentation.

---

# Setup

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Get a Gemini API Key

1. Go to: https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Create a new API key
4. Copy the key

---

## 3. Add the API Key to your environment

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

The code automatically loads this using `python-dotenv`.

---

## Alternative: set it in the terminal

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Then run the experiment:

```bash
python -m src.run_experiment
```

---

## Dataset

Experiments are conducted on the **Asia Bayesian Network**, a standard benchmark dataset used in causal discovery research.

The network contains the following variables:

- VisitAsia
- Smoking
- Tuberculosis
- LungCancer
- Bronchitis
- XRay
- Dyspnea

The ground-truth graph contains **8 directed causal edges**.

---

## Ground Truth Graph

The Asia Bayesian Network has the following causal structure:

```
VisitAsia → Tuberculosis
Smoking → LungCancer
Smoking → Bronchitis
Tuberculosis → Dyspnea
Tuberculosis → XRay
LungCancer → Dyspnea
LungCancer → XRay
Bronchitis → Dyspnea
```

A visual diagram of this network is commonly used as a benchmark for causal discovery algorithms.

---

---

## LLM Configuration

Experiments use the **Gemini 2.5 Flash** model.

Configuration details:

- Temperature: **0.0** (deterministic outputs)
- JSON-only responses enforced
- Automatic retry logic for API rate limits
- Disk caching enabled for repeated prompts

Caching ensures that repeated experiment runs reuse previous responses rather than making additional API calls.

---

## DAG Constraints

All discovered graphs must satisfy the **Directed Acyclic Graph (DAG)** constraint.

During graph construction:

- edges that introduce **cycles** are rejected
- duplicate edges are ignored

This guarantees that the resulting causal graph remains a valid causal structure.

---

## Structural Hamming Distance (SHD)

Structural Hamming Distance (SHD) measures the number of edge modifications required to transform the predicted graph into the true graph.

The following operations count toward SHD:

- inserting a missing edge
- deleting an incorrect edge
- reversing an incorrectly oriented edge

Lower SHD values indicate that the recovered graph is structurally closer to the ground-truth causal graph.

---

# Experimental Results (Asia Dataset)

We evaluate how different LLM interventions improve causal graph discovery on the **Asia Bayesian Network**, which contains **8 true causal edges**.

---

# 1. Baseline Causal Discovery

The baseline algorithm performs causal discovery using the **PC algorithm** on generated Asia data.

Procedure:

1. Generate samples from the Asia Bayesian Network.
2. Run the PC algorithm to recover a candidate graph.
3. Compare the predicted graph to the ground-truth graph.

### Result

The baseline algorithm correctly identifies **3 of the 8 true causal edges**.

Metrics:

- Precision: **0.75**
- Recall: **0.375**
- F1: **0.50**
- SHD: **6**

The baseline identifies some correct relationships but misses many true edges, resulting in low recall.

---

# 2. Intervention — LLM Edge Pruning

We apply an LLM verification step to remove edges that represent **indirect causal relationships**.

Function used:

```
verify_edge_direct(src, dst)
```

This step removes edges that are better explained by mediator variables.

### Result

- Precision: **1.00**
- Recall: **0.50**
- F1: **0.667**
- SHD: **4**

Edge pruning significantly improves precision by removing incorrect edges.

---

# 3. Intervention — Direction Correction

Next we verify whether discovered edges have the **correct causal direction**.

Function used:

```
verify_edge_direction(src, dst)
```

The model can:

- keep the edge
- flip the direction
- remove the edge

This step corrects incorrectly oriented causal relationships.

---

# 4. Intervention — Missing Edge Discovery

Finally we ask the LLM to inspect the current graph and propose **important missing causal edges**.

Function used:

```
suggest_missing_edges(current_edges, nodes)
```

The LLM suggests up to 5 additional edges that were not discovered during baseline causal discovery.

These edges are added if they preserve DAG structure.

---

# Final Pipeline Performance

| Method | Precision | Recall | F1 | SHD |
|------|------|------|------|------|
| Causal Baseline | 0.75 | 0.375 | 0.50 | 6 |
| + LLM Edge Pruning | 1.00 | 0.50 | 0.667 | 4 |
| + Direction Correction + Missing Edge Suggestions | 0.80 | 1.00 | **0.889** | **2** |

---

## LLM Usage Statistics

The system tracks LLM usage during experiments.

Example run statistics:

- LLM calls: **1**
- Cache hits: **10**

Because responses are cached locally, repeated runs require significantly fewer API calls. This makes experimentation faster and reduces API cost.

---

# Final Graph Recovery

---

## Recovered Graph

The final pipeline recovers the following edges:

```
Bronchitis → Dyspnea
LungCancer → Dyspnea
LungCancer → XRay
Smoking → Bronchitis
Smoking → LungCancer
Tuberculosis → Dyspnea
Tuberculosis → XRay
VisitAsia → Tuberculosis
```

Additional predicted edges:

```
Bronchitis → XRay
Smoking → Dyspnea
```

These two edges are not part of the ground‑truth network but were suggested by the LLM based on plausible medical relationships.

---

The full pipeline successfully recovers **all 8 true causal edges** in the Asia network.

Two additional edges are predicted:

```
Bronchitis → XRay
Smoking → Dyspnea
```

These are medically plausible relationships but are **not present in the ground-truth Asia network**, producing **2 false positives**.

Final counts:

- True Positives: **8**
- False Positives: **2**
- False Negatives: **0**

Final metrics:

- Precision = **0.80**
- Recall = **1.00**
- F1 = **0.889**
- Structural Hamming Distance (SHD) = **2**

---

# Key Takeaways

- LLM-guided **edge pruning** improves precision.
- **Direction correction** helps fix incorrectly oriented edges.
- **Missing-edge discovery** significantly improves recall.
- Combining these interventions produces the most accurate causal graph.

Overall, structured LLM interventions significantly improve causal graph recovery compared to the baseline discovery algorithm.

---

## Reproducing Results

To reproduce the experiment results:

```bash
python -m src.run_experiment
```

Results will be saved to the `results/` directory as JSON files containing:

- predicted edges
- ground-truth edges
- evaluation metrics
- LLM usage statistics

---

## Limitations

While LLM reasoning helps improve recall and edge direction accuracy, it can sometimes introduce **plausible but incorrect causal relationships** based on domain knowledge.  

For example, the model predicted:

- Bronchitis → XRay
- Smoking → Dyspnea

These relationships are medically plausible but are not present in the ground-truth Asia network, resulting in two false positives.

Future work could incorporate statistical testing or score-based validation to filter such edges.

---