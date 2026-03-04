# human-in-the-loop-causal-discovery
Replication and ablation study of LLM-guided causal graph discovery, evaluating where human intervention improves graph accuracy and efficiency.

## Project Structure

This project builds a causal graph using an LLM (Gemini) and a BFS-style search over variables.

---

### `src/run_experiment.py`

Entry point for running experiments.

Responsibilities:
- Loads the dataset (currently toy / Asia placeholder)
- Runs the BFS causal discovery algorithm
- Computes evaluation metrics (precision, recall, F1, SHD)
- Saves results to the `results/` folder

Run with:

```bash
python -m src.run_experiment
```

---

### `src/baseline_bfs.py`

Core algorithm for causal graph discovery.

Implements a BFS-style exploration of the causal graph.

Steps:
1. Ask the LLM for root variables
2. For each variable `X`, ask the LLM which variables `X` directly causes
3. Verify each proposed edge using the LLM
4. Add the edge only if it:
   - does not create a cycle
   - is verified as a direct causal relationship

This constructs the predicted directed acyclic graph (DAG).

---

### `src/llm_interface.py`

Interface for interacting with Gemini.

Handles:
- Prompt construction
- JSON-only responses
- Retry logic (including rate limits)
- API usage tracking
- Response caching

Main functions:
- `get_root_nodes()` – selects likely root causes
- `get_children(node)` – returns variables that the node may directly cause
- `verify_edge_direct(x, y)` – checks whether `x → y` is a direct causal relationship

---

### `src/cache.py`

Disk caching system for LLM responses.

Stores responses in `.cache_gemini` so repeated runs reuse previous answers instead of calling the API again.

This:
- speeds up experiments
- reduces API usage
- avoids rate limits

---

### `src/graph.py`

Simple directed graph implementation.

Handles:
- storing nodes and edges
- checking whether an edge already exists
- preventing cycles when adding edges

Ensures the graph remains a valid DAG.

---

### `src/logger.py`

Tracks experiment statistics and debugging information.

Logs:
- edges added
- edges skipped due to cycles
- duplicate edges
- messages during graph construction

Also stores counts used in the final experiment summary.

---

### `results/`

Stores experiment outputs.

Each run saves a JSON file containing:
- predicted edges
- ground truth edges
- precision, recall, and F1 score
- structural hamming distance (SHD)
- LLM usage statistics

---

### `.cache_gemini/`

Local cache for Gemini responses.

This folder stores cached LLM outputs so repeated runs can reuse previous answers without calling the API again. This significantly speeds up experimentation.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Get a Gemini API Key

1. Go to: https://aistudio.google.com/app/apikey  
2. Sign in with your Google account  
3. Create a new API key  
4. Copy the key

---

### 3. Add the API Key to your environment

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

The code automatically loads this using `python-dotenv`.

---

### Alternative (set it in the terminal)

If you do not want to use a `.env` file, you can export the key directly:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Then run the experiment:

```bash
python -m src.run_experiment
```