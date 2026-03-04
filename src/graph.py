from collections import defaultdict, deque
from typing import Iterable, Set, Tuple, Dict, List


Edge = Tuple[str, str]


class DiGraph:
    """
    Simple directed graph for causal edges.
    We keep both an edge set and adjacency list for fast checks.
    """

    def __init__(self, nodes: Iterable[str]):
        self.nodes: List[str] = list(nodes)
        self._adj: Dict[str, Set[str]] = defaultdict(set)
        self._edges: Set[Edge] = set()

    def add_edge(self, u: str, v: str) -> None:
        """Add directed edge u -> v (assumes you already checked validity)."""
        self._adj[u].add(v)
        self._edges.add((u, v))

    def has_edge(self, u: str, v: str) -> bool:
        return (u, v) in self._edges

    def edges(self) -> Set[Edge]:
        return set(self._edges)

    def would_create_cycle(self, u: str, v: str) -> bool:
        """
        Check if adding u -> v would create a cycle.
        This happens if v can already reach u in the current graph.
        """
        if u == v:
            return True

        # BFS from v to see if we can reach u
        q = deque([v])
        seen = {v}

        while q:
            x = q.popleft()
            if x == u:
                return True
            for nxt in self._adj.get(x, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)

        return False