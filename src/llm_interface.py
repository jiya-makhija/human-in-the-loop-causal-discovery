import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from google import genai

from .cache import DiskCache


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Gemini sometimes wraps JSON in extra text.
    Extract the first {...} block.
    """
    text = (text or "").strip()
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"No JSON found. Output was: {text[:200]}")
    return json.loads(m.group(0))


@dataclass
class LLMUsage:
    calls: int = 0
    cache_hits: int = 0


class GeminiLLM:
    """
    Gemini wrapper for:
      - stable prompts
      - JSON-only responses
      - disk caching
      - basic retries
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        cache_dir: str = ".cache_gemini",
        temperature: float = 0.2,
        max_output_tokens: int = 256,
        api_key_env: str = "GEMINI_API_KEY",
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {api_key_env}. Export it in your shell.")

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        self.cache = DiskCache(cache_dir)
        self.usage = LLMUsage()

    def _rules(self) -> str:
        return (
            "You are a careful causal discovery assistant.\n"
            "Return ONLY valid JSON. No prose.\n"
            "Only use variable names exactly as provided.\n"
            "Prefer direct causal effects, not correlation.\n"
            "If uncertain, return an empty list.\n"
        )

    def _call_json(self, prompt: str, cache_key: str, retries: int = 3) -> Dict[str, Any]:
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.usage.cache_hits += 1
            return cached

        last_err: Optional[Exception] = None

        for attempt in range(retries):
            try:
                self.usage.calls += 1
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_output_tokens,
                    },
                )
                data = _extract_json(resp.text)
                self.cache.set(cache_key, data)
                return data
            except Exception as e:
                last_err = e
                time.sleep(0.6 * (attempt + 1))

        raise RuntimeError(f"Gemini failed after retries: {last_err}")

    def get_root_nodes(self, nodes: List[str], descriptions: Optional[Dict[str, str]] = None) -> List[str]:
        desc_block = ""
        if descriptions:
            pairs = [f"- {k}: {descriptions.get(k, '')}" for k in nodes]
            desc_block = "\nVARIABLE DESCRIPTIONS:\n" + "\n".join(pairs) + "\n"

        prompt = (
            self._rules()
            + "\nTASK: Select 1–5 plausible ROOT causes.\n"
            + "Output JSON: {\"roots\": [\"Var1\", \"Var2\"]}\n"
            + f"\nVARIABLES:\n{nodes}\n"
            + desc_block
        )

        key = self.cache.key_for("roots", {"model": self.model, "nodes": nodes, "descriptions": bool(descriptions)})
        data = self._call_json(prompt, key)
        roots = data.get("roots", [])
        roots = [r for r in roots if isinstance(r, str) and r in nodes]
        return roots[:5]

    def get_children(self, node: str, nodes: List[str], descriptions: Optional[Dict[str, str]] = None) -> List[str]:
        desc_block = ""
        if descriptions:
            pairs = [f"- {k}: {descriptions.get(k, '')}" for k in nodes]
            desc_block = "\nVARIABLE DESCRIPTIONS:\n" + "\n".join(pairs) + "\n"

        prompt = (
            self._rules()
            + f"\nTASK: List 0–8 DIRECT effects of '{node}'.\n"
            + "Output JSON: {\"children\": [\"VarA\", \"VarB\"]}\n"
            + f"\nGIVEN NODE:\n{node}\n"
            + f"\nVARIABLES:\n{nodes}\n"
            + desc_block
        )

        key = self.cache.key_for(
            "children",
            {"model": self.model, "node": node, "nodes": nodes, "descriptions": bool(descriptions)},
        )
        data = self._call_json(prompt, key)
        children = data.get("children", [])
        children = [c for c in children if isinstance(c, str) and c in nodes and c != node]
        return children[:8]