from collections import deque
from typing import Dict, List, Optional

from .graph import DiGraph
from .logger import RunLogger
from .llm_interface import GeminiLLM


def build_graph_bfs(
    nodes: List[str],
    llm: GeminiLLM,
    descriptions: Optional[Dict[str, str]] = None,
    logger: Optional[RunLogger] = None,
    max_nodes_to_expand: Optional[int] = None,
) -> DiGraph:
    """
    BFS-style LLM causal graph discovery baseline.

    Steps:
      1) ask LLM for root nodes
      2) BFS expand: for each node X, ask which variables X causes
      3) add edges only if they do not create a cycle
    """

    logger = logger or RunLogger(verbose=False)
    G = DiGraph(nodes)

    roots = llm.get_root_nodes(nodes, descriptions=descriptions)
    logger.log(f"[roots] {roots}")

    q = deque(roots)
    visited = set()

    while q:
        x = q.popleft()
        if x in visited:
            continue
        visited.add(x)

        if max_nodes_to_expand is not None and len(visited) > max_nodes_to_expand:
            logger.log("[stop] hit max_nodes_to_expand")
            break

        children = llm.get_children(x, nodes, descriptions=descriptions)
        logger.log(f"[children] {x} -> {children}")

        for y in children:
            if G.has_edge(x, y):
                logger.edges_skipped_dup += 1
                continue

            if G.would_create_cycle(x, y):
                logger.edges_skipped_cycle += 1
                logger.log(f"[skip-cycle] {x} -> {y}")
                continue

            G.add_edge(x, y)
            logger.edges_added += 1
            logger.log(f"[add] {x} -> {y}")

            if y not in visited:
                q.append(y)

    return G