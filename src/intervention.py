from typing import Any, Dict, List, Set, Tuple

from .llm_interface import GeminiLLM

Edge = Tuple[str, str]


def prune_indirect_edges(
    edges: Set[Edge],
    nodes: List[str],
    llm: GeminiLLM,
    descriptions: Dict[str, str] | None = None,
    verbose: bool = True,
) -> Set[Edge]:
    """
    Remove edges that are indirect or implausible.
    """
    pruned_edges = set()

    for src, dst in edges:
        verdict = llm.verify_edge_direct(
            src,
            dst,
            nodes,
            descriptions=descriptions,
        )

        if verdict.get("keep", False):
            pruned_edges.add((src, dst))
            if verbose:
                print(f"[keep] {src} -> {dst}")
        else:
            mediators = verdict.get("mediators", [])
            if verbose:
                if mediators:
                    print(f"[remove] {src} -> {dst} (mediators={mediators})")
                else:
                    print(f"[remove] {src} -> {dst}")

    return pruned_edges


def correct_edge_directions(
    edges: Set[Edge],
    nodes: List[str],
    llm: GeminiLLM,
    descriptions: Dict[str, str] | None = None,
    verbose: bool = True,
) -> Set[Edge]:
    """
    Post-processing intervention step that checks whether each edge
    should be kept, flipped, or removed.

    This is useful for correcting direction mistakes after the baseline
    causal graph has already been constructed.
    """
    corrected_edges = set()

    for src, dst in edges:
        verdict = llm.verify_edge_direction(
            src,
            dst,
            nodes,
            descriptions=descriptions,
        )

        action = verdict.get("action", "keep")
        reason = verdict.get("reason", "")

        if action == "flip":
            corrected_edges.add((dst, src))
            if verbose:
                if reason:
                    print(f"[flip] {src} -> {dst}  =>  {dst} -> {src} ({reason})")
                else:
                    print(f"[flip] {src} -> {dst}  =>  {dst} -> {src}")

        elif action == "remove":
            if verbose:
                if reason:
                    print(f"[remove] {src} -> {dst} ({reason})")
                else:
                    print(f"[remove] {src} -> {dst}")

        else:
            corrected_edges.add((src, dst))
            if verbose:
                if reason:
                    print(f"[keep] {src} -> {dst} ({reason})")
                else:
                    print(f"[keep] {src} -> {dst}")

    return corrected_edges

def suggest_missing_edges(
    edges: Set[Edge],
    nodes: List[str],
    llm: GeminiLLM,
    descriptions: Dict[str, str] | None = None,
    max_edges: int = 5,
    verbose: bool = True,
) -> Set[Edge]:
    """
    Ask the LLM for a small number of missing direct causal edges
    and add them to the current graph.
    """
    current_edges = sorted(list(edges))

    verdict = llm.suggest_missing_edges(
        current_edges=current_edges,
        nodes=nodes,
        descriptions=descriptions,
        max_edges=max_edges,
    )

    suggested_edges = verdict.get("suggested_edges", [])
    updated_edges = set(edges)

    for src, dst in suggested_edges:
        if (src, dst) not in updated_edges:
            updated_edges.add((src, dst))
            if verbose:
                print(f"[add-missing] {src} -> {dst}")

    return updated_edges