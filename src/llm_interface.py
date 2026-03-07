import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# load .env automatically so small scripts work too
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from google import genai

from .cache import DiskCache


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Parse a JSON object from Gemini output.
    Handles markdown fences, extra text, and partial JSON cases.
    """
    if text is None:
        raise ValueError("No text returned from model")

    s = text.strip()

    # remove markdown code fences if present
    if s.startswith("```"):
        lines = s.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    # try parsing the whole string first
    try:
        return json.loads(s)
    except Exception:
        pass

    # fallback: extract first JSON-looking block
    m = _JSON_RE.search(s)
    if not m:
        if "{" in s and "}" not in s:
            raise ValueError(f"Truncated JSON from model: {s[:200]}")
        raise ValueError(f"No JSON found. Output was: {s[:200]}")

    block = m.group(0)
    try:
        return json.loads(block)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON block: {block[:200]}") from e


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
    - retry logic
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        cache_dir: str = ".cache_gemini",
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
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
            "Do NOT wrap JSON in Markdown fences (no ```json).\n"
            "Only use variable names exactly as provided.\n"
            "Prefer direct causal effects, not correlation.\n"
            "If unsure, return an empty list.\n"
        )

    def _call_json(self, prompt: str, cache_key: str, retries: int = 10) -> Dict[str, Any]:
        """
        Call Gemini and return parsed JSON.

        - uses disk cache first
        - retries on parse errors
        - retries on 429 / RESOURCE_EXHAUSTED
        """
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.usage.cache_hits += 1
            return cached

        last_err: Optional[Exception] = None
        retry_suffixes = [
            "",
            "\n\nIMPORTANT: Your last response was invalid or incomplete. Return ONE complete JSON object ONLY. No markdown. No extra text.",
            "\n\nFINAL REMINDER: Output must be valid JSON and must include all closing brackets/braces.",
        ]

        def _sleep_for_rate_limit(err_text: str, attempt: int) -> None:
            m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", err_text, re.IGNORECASE)
            if m:
                delay = float(m.group(1))
            else:
                m2 = re.search(r"retryDelay\"\s*:\s*\"([0-9]+)s\"", err_text)
                delay = float(m2.group(1)) if m2 else 0.0

            if delay <= 0:
                delay = min(60.0, 2.0 * (attempt + 1))

            print(f"[rate-limit] sleeping {delay + 1.0:.1f}s")
            time.sleep(delay + 1.0)

        for attempt in range(retries):
            try:
                self.usage.calls += 1
                tightened_prompt = prompt + retry_suffixes[min(attempt, len(retry_suffixes) - 1)]

                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=tightened_prompt,
                    config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_output_tokens,
                        "response_mime_type": "application/json",
                    },
                )

                data = _extract_json(resp.text)
                self.cache.set(cache_key, data)
                return data

            except Exception as e:
                last_err = e
                err_text = str(e)

                if "RESOURCE_EXHAUSTED" in err_text or "429" in err_text:
                    _sleep_for_rate_limit(err_text, attempt)
                    continue

                time.sleep(0.6 * (attempt + 1))

        raise RuntimeError(f"Gemini failed after retries: {last_err}")

    def get_root_nodes(
        self,
        nodes: List[str],
        descriptions: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        desc_block = ""
        if descriptions:
            pairs = [f"- {k}: {descriptions.get(k, '')}" for k in nodes]
            desc_block = "\nVARIABLE DESCRIPTIONS:\n" + "\n".join(pairs) + "\n"

        prompt = (
            self._rules()
            + "\nTASK: Select 1–5 plausible ROOT causes.\n"
            + 'Output EXACTLY one JSON object (no Markdown): {"roots": ["Var1", "Var2"]}\n'
            + f"\nVARIABLES:\n{nodes}\n"
            + desc_block
        )

        key = self.cache.key_for(
            "roots",
            {
                "model": self.model,
                "nodes": nodes,
                "descriptions": bool(descriptions),
            },
        )

        data = self._call_json(prompt, key)
        roots = data.get("roots", [])
        roots = [r for r in roots if isinstance(r, str) and r in nodes]
        return roots[:5]

    def get_children(
        self,
        node: str,
        nodes: List[str],
        descriptions: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        desc_block = ""
        if descriptions:
            pairs = [f"- {k}: {descriptions.get(k, '')}" for k in nodes]
            desc_block = "\nVARIABLE DESCRIPTIONS:\n" + "\n".join(pairs) + "\n"

        prompt = (
            self._rules()
            + f"\nTASK: List 0–8 DIRECT effects of '{node}'.\n"
            + 'Output EXACTLY one JSON object (no Markdown): {"children": ["VarA", "VarB"]}\n'
            + f"\nGIVEN NODE:\n{node}\n"
            + f"\nVARIABLES:\n{nodes}\n"
            + desc_block
        )

        key = self.cache.key_for(
            "children",
            {
                "model": self.model,
                "node": node,
                "nodes": nodes,
                "descriptions": bool(descriptions),
            },
        )

        data = self._call_json(prompt, key)
        children = data.get("children", [])
        children = [c for c in children if isinstance(c, str) and c in nodes and c != node]
        return children[:8]

    def verify_edge_direct(
        self,
        src: str,
        dst: str,
        nodes: List[str],
        descriptions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Verify whether src -> dst is a direct causal edge.
        """
        if src not in nodes or dst not in nodes or src == dst:
            return {"keep": False, "mediators": [], "reason": "invalid variables"}

        desc_block = ""
        if descriptions:
            pairs = [f"- {k}: {descriptions.get(k, '')}" for k in nodes]
            desc_block = "\nVARIABLE DESCRIPTIONS:\n" + "\n".join(pairs) + "\n"

        prompt = (
            self._rules()
            + "\nTASK: Decide if the edge is DIRECT (not explained by a mediator among the variables).\n"
            + f"EDGE CANDIDATE: '{src}' -> '{dst}'\n"
            + 'Return EXACTLY one JSON object (no Markdown): {"keep": true, "mediators": ["Var"], "reason": "one short sentence"}'
            + "\n\nGuidelines:\n"
            + "- keep=true ONLY if src is a direct cause of dst given the listed variables.\n"
            + "- If the effect is indirect, set keep=false and list 0-3 mediators.\n"
            + "- If unsure, set keep=false.\n"
            + f"\nVARIABLES:\n{nodes}\n"
            + desc_block
        )

        key = self.cache.key_for(
            "verify_edge_direct",
            {
                "model": self.model,
                "src": src,
                "dst": dst,
                "nodes": nodes,
                "descriptions": bool(descriptions),
            },
        )

        data = self._call_json(prompt, key)

        keep = bool(data.get("keep", False))

        mediators = data.get("mediators", [])
        if not isinstance(mediators, list):
            mediators = []
        mediators = [
            m for m in mediators
            if isinstance(m, str) and m in nodes and m not in (src, dst)
        ][:3]

        reason = data.get("reason", "")
        if not isinstance(reason, str):
            reason = ""
        reason = reason.strip()[:180]

        return {
            "keep": keep,
            "mediators": mediators,
            "reason": reason,
        }

    def verify_edge_direction(
        self,
        src: str,
        dst: str,
        nodes: List[str],
        descriptions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Decide whether an edge should be kept, flipped, or removed.
        """
        if src not in nodes or dst not in nodes or src == dst:
            return {"action": "remove", "reason": "invalid variables"}

        desc_block = ""
        if descriptions:
            pairs = [f"- {k}: {descriptions.get(k, '')}" for k in nodes]
            desc_block = "\nVARIABLE DESCRIPTIONS:\n" + "\n".join(pairs) + "\n"

        prompt = (
            self._rules()
            + "\nTASK: Decide whether the direction of the candidate edge is correct.\n"
            + f"CANDIDATE EDGE: '{src}' -> '{dst}'\n"
            + 'Return EXACTLY one JSON object (no Markdown): {"action": "keep", "reason": "one short sentence"}'
            + "\n\nAllowed actions:\n"
            + "- keep: if src -> dst is the more plausible direction\n"
            + "- flip: if dst -> src is more plausible\n"
            + "- remove: if neither direction is a good direct causal edge\n"
            + "- If unsure, choose remove.\n"
            + f"\nVARIABLES:\n{nodes}\n"
            + desc_block
        )

        key = self.cache.key_for(
            "verify_edge_direction",
            {
                "model": self.model,
                "src": src,
                "dst": dst,
                "nodes": nodes,
                "descriptions": bool(descriptions),
            },
        )

        data = self._call_json(prompt, key)

        action = data.get("action", "remove")
        if not isinstance(action, str):
            action = "remove"
        action = action.strip().lower()
        if action not in {"keep", "flip", "remove"}:
            action = "remove"

        reason = data.get("reason", "")
        if not isinstance(reason, str):
            reason = ""
        reason = reason.strip()[:180]

        return {
            "action": action,
            "reason": reason,
        }

    def suggest_missing_edges(
        self,
        current_edges: List[tuple[str, str]],
        nodes: List[str],
        descriptions: Optional[Dict[str, str]] = None,
        max_edges: int = 5,
    ) -> Dict[str, Any]:
        """
        Suggest a small number of missing direct causal edges.
        """
        desc_block = ""
        if descriptions:
            pairs = [f"- {k}: {descriptions.get(k, '')}" for k in nodes]
            desc_block = "\nVARIABLE DESCRIPTIONS:\n" + "\n".join(pairs) + "\n"

        prompt = (
            self._rules()
            + "\nTASK: Suggest a small number of important DIRECT causal edges that are missing from the current graph.\n"
            + "Only suggest edges between the listed variables.\n"
            + "Do NOT repeat edges already present.\n"
            + f"Suggest at most {max_edges} edges.\n"
            + 'Return EXACTLY one JSON object (no Markdown): {"suggested_edges": [["A","B"]], "reason": "one short sentence"}\n'
            + f"\nVARIABLES:\n{nodes}\n"
            + f"\nCURRENT EDGES:\n{current_edges}\n"
            + desc_block
        )

        key = self.cache.key_for(
            "suggest_missing_edges",
            {
                "model": self.model,
                "nodes": nodes,
                "current_edges": current_edges,
                "descriptions": bool(descriptions),
                "max_edges": max_edges,
            },
        )

        data = self._call_json(prompt, key)

        suggested = data.get("suggested_edges", [])
        if not isinstance(suggested, list):
            suggested = []

        cleaned = []
        current_edge_set = {tuple(edge) for edge in current_edges}

        for edge in suggested:
            if not isinstance(edge, list) or len(edge) != 2:
                continue

            src, dst = edge
            if not isinstance(src, str) or not isinstance(dst, str):
                continue
            if src not in nodes or dst not in nodes or src == dst:
                continue
            if (src, dst) in current_edge_set:
                continue

            cleaned.append((src, dst))

        cleaned = cleaned[:max_edges]

        reason = data.get("reason", "")
        if not isinstance(reason, str):
            reason = ""
        reason = reason.strip()[:180]

        return {
            "suggested_edges": cleaned,
            "reason": reason,
        }