import hashlib
import json
import os
from typing import Any, Dict, Optional


class DiskCache:
    """
    Tiny JSON disk cache.
    Saves model outputs so reruns don't repeat API calls.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def key_for(self, kind: str, payload: Dict[str, Any]) -> str:
        raw = json.dumps({"kind": kind, "payload": payload}, sort_keys=True).encode("utf-8")
        h = hashlib.sha256(raw).hexdigest()
        return f"{kind}_{h}"

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        path = self._path(key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)