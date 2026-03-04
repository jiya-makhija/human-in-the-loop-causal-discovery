from typing import Dict, Set, Tuple

Edge = Tuple[str, str]


def edge_f1(pred_edges: Set[Edge], true_edges: Set[Edge]) -> Dict[str, float]:
    """
    Precision/Recall/F1 for directed edges.
    """
    pred_edges = set(pred_edges)
    true_edges = set(true_edges)

    tp = len(pred_edges & true_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def shd_directed(pred_edges: Set[Edge], true_edges: Set[Edge]) -> int:
    """
    Directed SHD (simple):
      (# extra edges) + (# missing edges)
    """
    pred_edges = set(pred_edges)
    true_edges = set(true_edges)
    return len(pred_edges - true_edges) + len(true_edges - pred_edges)