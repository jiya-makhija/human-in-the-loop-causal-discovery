import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Load .env automatically so running small scripts works too
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from google import genai

from .cache import DiskCache


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse a JSON object from Gemini output.

    Handles:
      - Markdown fences like ```json ... ```
      - Extra text around JSON
      - Truncated JSON (started but not finished)

    Returns a dict or raises ValueError.
    """
    if text is None:
        raise ValueError("No text returned from model")

    s = text.strip()

    # Strip Markdown code fences if present
    if s.startswith("```"):
        lines = s.splitlines()
        if lines:
            lines = lines[1:]  # drop ``` or ```json
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    # First: try parsing the whole string
    try:
        return json.loads(s)
    except Exception:
        pass

    # Fallback: extract a JSON object block
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
      - basic retries
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
            "If uncertain, return an empty list.\n"
        )

    def _call_json(self, prompt: str, cache_key: str, retries: int = 10) -> Dict[str, Any]:
        """Call Gemini and return parsed JSON.

        - Uses disk cache first.
        - Retries on parse errors.
        - Retries on 429 RESOURCE_EXHAUSTED by sleeping for the suggested delay.
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
            # Common formats:
            #   "Please retry in 8.73s."
            #   "retryDelay": "8s"
            m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", err_text, re.IGNORECASE)
            if m:
                delay = float(m.group(1))
            else:
                m2 = re.search(r"retryDelay\"\s*:\s*\"([0-9]+)s\"", err_text)
                delay = float(m2.group(1)) if m2 else 0.0

            if delay <= 0:
                # fallback backoff (cap at 60s)
                delay = min(60.0, 2.0 * (attempt + 1))

            # add buffer so we clear the rate-limit window
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

                # Handle free-tier RPM limits
                if "RESOURCE_EXHAUSTED" in err_text or "429" in err_text:
                    _sleep_for_rate_limit(err_text, attempt)
                    continue

                # Otherwise: small backoff and retry
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
            + "Output EXACTLY one JSON object (no Markdown): {\"roots\": [\"Var1\", \"Var2\"]}\n"
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
            + "Output EXACTLY one JSON object (no Markdown): {\"children\": [\"VarA\", \"VarB\"]}\n"
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

    def verify_edge_direct(
        self,
        src: str,
        dst: str,
        nodes: List[str],
        descriptions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Verify whether src -> dst is a DIRECT causal edge (not mediated).

        Returns a dict like:
          {"keep": true/false, "mediators": ["Var"], "reason": "..."}

        Notes:
        - Uses caching via _call_json.
        - Filters mediators to variables present in `nodes`.
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
            + "Return EXACTLY one JSON object (no Markdown): "
            + '{"keep": true, "mediators": ["Var"], "reason": "one short sentence"}'
            + "\n\nGuidelines:\n"
            + "- keep=true ONLY if src is a direct cause of dst given the listed variables.\n"
            + "- If the effect is indirect, set keep=false and list 0-3 mediators (variables that explain the link).\n"
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
        mediators = [m for m in mediators if isinstance(m, str) and m in nodes and m not in (src, dst)]
        mediators = mediators[:3]

        reason = data.get("reason", "")
        if not isinstance(reason, str):
            reason = ""
        reason = reason.strip()
        if len(reason) > 180:
            reason = reason[:180]

        return {"keep": keep, "mediators": mediators, "reason": reason}