from collections import defaultdict, deque
from typing import Dict, Iterable, List, Set, Tuple

Edge = Tuple[str, str]


class DiGraph:
    """
    Simple directed graph for causal edges.
    Provides cycle check to enforce DAG constraint.
    """

    def __init__(self, nodes: Iterable[str]):
        self.nodes: List[str] = list(nodes)
        self._adj: Dict[str, Set[str]] = defaultdict(set)
        self._edges: Set[Edge] = set()

    def add_edge(self, u: str, v: str) -> None:
        """Add edge u -> v."""
        self._adj[u].add(v)
        self._edges.add((u, v))

    def has_edge(self, u: str, v: str) -> bool:
        return (u, v) in self._edges

    def edges(self) -> Set[Edge]:
        return set(self._edges)

    def would_create_cycle(self, u: str, v: str) -> bool:
        """
        Adding u -> v creates a cycle if v can already reach u.
        """
        if u == v:
            return True

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