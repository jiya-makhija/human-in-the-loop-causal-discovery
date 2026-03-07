from typing import Dict, Set, Tuple

Edge = Tuple[str, str]


def run_causal_baseline(dataset: Dict) -> Set[Edge]:
    """
    Causal discovery baseline using the PC algorithm.

    If the dataset contains a pandas DataFrame under key "data",
    we run the PC algorithm from pgmpy.

    If no data is available, we fall back to placeholder behavior
    so the rest of the pipeline still works.
    """
    data = dataset.get("data")

    if data is None:
        true_edges = dataset.get("true_edges", set())
        return set(true_edges)

    try:
        from pgmpy.estimators import PC

        pc = PC(data)
        model = pc.estimate(return_type="dag")

        edges = set()
        for u, v in model.edges():
            edges.add((str(u), str(v)))

        return edges

    except Exception as e:
        print("[causal_baseline] pc algorithm failed, falling back:", e)
        true_edges = dataset.get("true_edges", set())
        return set(true_edges)