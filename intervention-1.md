# Intervention Step: LLM-Assisted Edge Verification

After running a causal discovery baseline (PC algorithm), we apply a **post-processing intervention step** to improve the predicted graph.

The goal of this step is to identify **edges that are not direct causal relationships**, but instead occur because of **mediated or indirect effects**.

---

# Motivation

Standard causal discovery algorithms such as:

- PC
- GES
- NOTEARS
- DAG-GNN

often produce graphs that contain:

- false positives  
- indirect relationships  
- direction mistakes  

For example, an algorithm might predict:

Smoking → Dyspnea

However, in the true causal structure the relationship is:

Smoking → Bronchitis → Dyspnea

So the predicted edge is **indirect**, not a direct causal effect.

Our intervention step uses an LLM to detect these situations.

---

# Method

After the baseline graph is produced, we iterate over each predicted edge.

For each edge:

X → Y

we ask the LLM:

> Is this a **direct causal relationship**, or is it explained by another variable in the graph?

The LLM returns a structured response such as:

```json
{
  "keep": true,
  "mediators": []
}
```

or

```json
{
  "keep": false,
  "mediators": ["Bronchitis"]
}
```

If `keep = false`, the edge is removed from the graph.

---

# Implementation

This step is implemented in:

```
src/intervention.py
```

Function:

```
prune_indirect_edges()
```

Pseudo-logic:

```
for each predicted edge (X → Y):

    ask LLM if edge is direct

    if direct:
        keep edge

    if mediated:
        remove edge
```

---

# Pipeline

The full experiment pipeline becomes:

```
Dataset
   ↓
PC causal discovery
   ↓
Predicted graph
   ↓
LLM edge verification
   ↓
Pruned graph
   ↓
Evaluation
```

---

# Example Result (Asia Network)

### Baseline (PC)

| metric | value |
|------|------|
| precision | 0.75 |
| recall | 0.375 |
| F1 | 0.50 |
| SHD | 6 |

False positive example:

```
Bronchitis → Smoking
```

---

### After Intervention

| metric | value |
|------|------|
| precision | **1.0** |
| recall | **0.5** |
| F1 | **0.667** |
| SHD | **4** |

The intervention removed the incorrect edge while keeping valid ones.

---

# Cost

The intervention required only:

```
4 LLM calls
```

This demonstrates that **small amounts of reasoning intervention can improve causal discovery outputs without replacing the underlying algorithm.**


Instead of querying the LLM over all possible variable pairs (O(n²) queries), we restrict LLM reasoning to only edges predicted by a causal discovery algorithm. In the Asia network experiment (7 variables), this reduced the required number of LLM calls from ~42 potential edge checks to only 4 verification queries.

intervention 2: llm edge direction correction

right now your first intervention only does:
	•	remove bad edges
	•	helps precision

but it does not recover missing edges or fix reversed edges.

a second intervention can do:
	•	check whether a predicted edge has the right direction
	•	if direction looks wrong, flip it

that helps recall and sometimes SHD too.