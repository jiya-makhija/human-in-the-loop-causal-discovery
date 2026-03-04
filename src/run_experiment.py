import json
import os
from datetime import datetime
from typing import Dict

from dotenv import load_dotenv

from .baseline_bfs import build_graph_bfs
from .datasets import load_asia_placeholder, load_toy
from .llm_interface import GeminiLLM
from .logger import RunLogger
from .metrics import edge_f1, shd_directed


load_dotenv()


def run_one(dataset: Dict, model: str = "gemini-2.5-flash", verbose: bool = True) -> Dict:
    llm = GeminiLLM(model=model)
    logger = RunLogger(verbose=verbose)

    nodes = dataset["nodes"]
    descriptions = dataset.get("descriptions")
    true_edges = dataset.get("true_edges", set())

    G = build_graph_bfs(
        nodes=nodes,
        llm=llm,
        descriptions=descriptions,
        logger=logger,
    )

    pred_edges = G.edges()

    metrics = {
        "dataset": dataset["name"],
        "model": model,
        "n_nodes": len(nodes),
        "n_true_edges": len(true_edges),
        "n_pred_edges": len(pred_edges),
        "f1": edge_f1(pred_edges, true_edges),
        "shd": shd_directed(pred_edges, true_edges),
        "llm_calls": llm.usage.calls,
        "cache_hits": llm.usage.cache_hits,
        "edges_added": logger.edges_added,
        "edges_skipped_cycle": logger.edges_skipped_cycle,
        "edges_skipped_dup": logger.edges_skipped_dup,
    }

    return metrics


def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Set GEMINI_API_KEY before running.")

    # Choose dataset here
    dataset = load_toy()
    # dataset = load_asia_placeholder()

    out = run_one(dataset, model="gemini-2.5-flash", verbose=True)

    print("\n=== SUMMARY ===")
    print(json.dumps(out, indent=2))

    # Save results
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results/{out['dataset']}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()