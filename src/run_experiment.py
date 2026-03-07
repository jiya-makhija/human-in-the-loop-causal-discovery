import json
import os
from datetime import datetime
from typing import Dict, Set, Tuple

from dotenv import load_dotenv

from .baseline_bfs import build_graph_bfs
from .causal_baseline import run_causal_baseline
from .datasets import load_asia_generated, load_asia_placeholder, load_toy
from .intervention import (
    prune_indirect_edges,
    correct_edge_directions,
    suggest_missing_edges,
)
from .llm_interface import GeminiLLM
from .logger import RunLogger
from .metrics import edge_f1, shd_directed

load_dotenv()

Edge = Tuple[str, str]


def summarize_result(
    dataset_name: str,
    method_name: str,
    pred_edges: Set[Edge],
    true_edges: Set[Edge],
    llm_calls: int = 0,
    cache_hits: int = 0,
) -> Dict:
    pred_edges_set = set(pred_edges)
    true_edges_set = set(true_edges)

    fp_edges = sorted(list(pred_edges_set - true_edges_set))
    fn_edges = sorted(list(true_edges_set - pred_edges_set))

    return {
        "dataset": dataset_name,
        "method": method_name,
        "n_true_edges": len(true_edges_set),
        "n_pred_edges": len(pred_edges_set),
        "pred_edges": sorted(list(pred_edges_set)),
        "true_edges": sorted(list(true_edges_set)),
        "fp_edges": fp_edges,
        "fn_edges": fn_edges,
        "f1": edge_f1(pred_edges_set, true_edges_set),
        "shd": shd_directed(pred_edges_set, true_edges_set),
        "llm_calls": llm_calls,
        "cache_hits": cache_hits,
    }


def run_bfs_baseline(dataset: Dict) -> Dict:
    llm = GeminiLLM(model="gemini-2.5-flash")
    logger = RunLogger(verbose=True)

    nodes = dataset["nodes"]
    descriptions = dataset.get("descriptions")
    true_edges = set(dataset.get("true_edges", set()))

    graph = build_graph_bfs(
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        logger=logger,
    )

    pred_edges = graph.edges()

    return summarize_result(
        dataset_name=dataset["name"],
        method_name="llm_bfs_baseline",
        pred_edges=pred_edges,
        true_edges=true_edges,
        llm_calls=llm.usage.calls,
        cache_hits=llm.usage.cache_hits,
    )


def run_standard_baseline(dataset: Dict) -> Dict:
    true_edges = set(dataset.get("true_edges", set()))
    pred_edges = run_causal_baseline(dataset)

    return summarize_result(
        dataset_name=dataset["name"],
        method_name="causal_baseline",
        pred_edges=pred_edges,
        true_edges=true_edges,
    )


def run_standard_plus_intervention(dataset: Dict) -> Dict:
    llm = GeminiLLM(model="gemini-2.5-flash")

    nodes = dataset["nodes"]
    descriptions = dataset.get("descriptions")
    true_edges = set(dataset.get("true_edges", set()))

    baseline_edges = run_causal_baseline(dataset)
    pruned_edges = prune_indirect_edges(
        edges=baseline_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        verbose=True,
    )

    return summarize_result(
        dataset_name=dataset["name"],
        method_name="causal_baseline_plus_pruning",
        pred_edges=pruned_edges,
        true_edges=true_edges,
        llm_calls=llm.usage.calls,
        cache_hits=llm.usage.cache_hits,
    )


def run_standard_plus_two_interventions(dataset: Dict) -> Dict:
    llm = GeminiLLM(model="gemini-2.5-flash")

    nodes = dataset["nodes"]
    descriptions = dataset.get("descriptions")
    true_edges = set(dataset.get("true_edges", set()))

    baseline_edges = run_causal_baseline(dataset)

    pruned_edges = prune_indirect_edges(
        edges=baseline_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        verbose=True,
    )

    corrected_edges = correct_edge_directions(
        edges=pruned_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        verbose=True,
    )

    return summarize_result(
        dataset_name=dataset["name"],
        method_name="causal_baseline_plus_pruning_plus_direction",
        pred_edges=corrected_edges,
        true_edges=true_edges,
        llm_calls=llm.usage.calls,
        cache_hits=llm.usage.cache_hits,
    )

def run_standard_plus_three_interventions(dataset: Dict) -> Dict:
    llm = GeminiLLM(model="gemini-2.5-flash")

    nodes = dataset["nodes"]
    descriptions = dataset.get("descriptions")
    true_edges = set(dataset.get("true_edges", set()))

    baseline_edges = run_causal_baseline(dataset)

    pruned_edges = prune_indirect_edges(
        edges=baseline_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        verbose=True,
    )

    corrected_edges = correct_edge_directions(
        edges=pruned_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        verbose=True,
    )

    completed_edges = suggest_missing_edges(
        edges=corrected_edges,
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        max_edges=5,
        verbose=True,
    )

    return summarize_result(
        dataset_name=dataset["name"],
        method_name="causal_baseline_plus_pruning_plus_direction_plus_missing",
        pred_edges=completed_edges,
        true_edges=true_edges,
        llm_calls=llm.usage.calls,
        cache_hits=llm.usage.cache_hits,
    )

def save_result(result: Dict) -> None:
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results/{result['dataset']}_{result['method']}_{ts}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nsaved -> {path}")


def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("set GEMINI_API_KEY before running.")

    # dataset = load_toy()
    # dataset = load_asia_placeholder()
    dataset = load_asia_generated(n_samples=2000)

    # choose one at a time
    # result = run_bfs_baseline(dataset)
    # result = run_standard_baseline(dataset)
    # result = run_standard_plus_intervention(dataset)
    # result = run_standard_plus_two_interventions(dataset)
    result = run_standard_plus_three_interventions(dataset)

    print("\n=== summary ===")
    print(json.dumps(result, indent=2))

    save_result(result)


if __name__ == "__main__":
    main()