"""
multi_agent.py — Two-agent collaboration pipeline.

  Agent 1 · Bùi Đức Toàn (Ph.D AI Research Scientist)
            Analyzes history → forms research hypothesis → proposes param change.

  Agent 2 · Vũ Thế Đệ   (Lead AI Engineer)
            Reviews proposal for engineering feasibility → implements final train.py.

Each step emits structured events via an optional `event_cb` callback so that
a UI (e.g. Gradio) can display real-time reasoning without polling.
"""

from __future__ import annotations

import json
import time
from typing import Callable

from agent.llm_client   import LLMClient
from agent.parser        import LLMParser
from agent.prompt_builder import PromptBuilder
from config              import Config


# ── Event helpers ─────────────────────────────────────────────────────────────

def _evt(agent: str, kind: str, content: str) -> dict:
    return {"ts": time.time(), "agent": agent, "kind": kind, "content": content}


# ── Scientist response parser ─────────────────────────────────────────────────

def _parse_scientist(raw: str) -> dict:
    """Parse Scientist JSON (no new_code, just structured proposal)."""
    text = raw.strip()
    # Strip outer markdown fence if present
    import re
    m = re.match(r"^```(?:json)?\s*\n(.*)\n```\s*$", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "proposed_param" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    # Fallback: extract first JSON object
    i = text.find("{")
    if i >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, i)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # Soft fallback — return raw as unstructured
    return {
        "analysis":       raw[:300],
        "hypothesis":     raw[:300],
        "proposed_param": "unknown",
        "proposed_value": "unknown",
        "confidence":     0.5,
        "reasoning":      raw,
    }


# ── Main coordinator ─────────────────────────────────────────────────────────

class MultiAgentCoordinator:
    """
    Orchestrates the Scientist→Engineer proposal pipeline.

    event_cb(event_dict) is called at each stage so callers can stream
    agent reasoning to a UI without blocking.
    """

    def __init__(
        self,
        scientist_model: str = Config.SCIENTIST_MODEL,
        engineer_model:  str = Config.ENGINEER_MODEL,
        event_cb: Callable[[dict], None] | None = None,
    ):
        self.scientist = LLMClient(scientist_model)
        self.engineer  = LLMClient(engineer_model)
        self.cb = event_cb or (lambda e: None)

        # Human-readable names for logging
        self._sci = Config.SCIENTIST_NAME
        self._eng = Config.ENGINEER_NAME

    # ── Public API ────────────────────────────────────────────────────────

    def propose(self, history: list[dict], current_code: str) -> dict:
        """
        Run the two-agent pipeline and return a dict compatible with the
        existing pipeline:  {description, new_code, scientist_proposal}
        """
        # ── Step 1: Scientist proposes ────────────────────────────────────
        self.cb(_evt(self._sci, "status",   "🔬 Analyzing experiment history..."))
        sys_p, usr_p = PromptBuilder.build_scientist(history, current_code)
        self.cb(_evt(self._sci, "prompt",   usr_p))

        sci_raw = self.scientist.generate(sys_p, usr_p)
        self.cb(_evt(self._sci, "response", sci_raw))

        proposal = _parse_scientist(sci_raw)
        self.cb(_evt(self._sci, "parsed",   json.dumps(proposal, ensure_ascii=False, indent=2)))

        # ── Step 2: Engineer reviews & implements ─────────────────────────
        self.cb(_evt(self._eng, "status",   "⚙️  Reviewing proposal and implementing code..."))
        sys_e, usr_e = PromptBuilder.build_engineer(history, current_code, proposal)
        self.cb(_evt(self._eng, "prompt",   usr_e))

        eng_raw = self.engineer.generate(sys_e, usr_e)
        self.cb(_evt(self._eng, "response", eng_raw))

        result = LLMParser.parse(eng_raw)       # → {description, new_code}
        result["scientist_proposal"] = proposal
        result["engineer_raw"]       = eng_raw

        # Log engineer review separately if present
        try:
            eng_obj = json.JSONDecoder().raw_decode(eng_raw.strip(), eng_raw.strip().find("{"))[0]
            review  = eng_obj.get("review", "")
            refine  = eng_obj.get("refinement", "")
            if review or refine:
                self.cb(_evt(self._eng, "review", f"Review: {review}\nRefinement: {refine}"))
        except Exception:
            pass

        self.cb(_evt(self._eng, "status", f"✅ Proposal accepted — {result['description']}"))
        return result