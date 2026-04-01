import re
from config import Config


class PromptBuilder:
    """Builds prompts for both the Scientist and Engineer agents."""

    # ── Shared context builder ────────────────────────────────────────────
    @staticmethod
    def _context_block(history: list[dict], current_code: str) -> dict:
        kept      = [h for h in history if h["status"] == "keep"]
        crashed   = [h for h in history if h["status"] == "crash"]
        discarded = [h for h in history if h["status"] == "discard"]
        baseline  = history[0]["val_mAP5095"] if history else 0.0
        best_map  = max((h["val_mAP5095"] for h in history), default=0.0)
        recent    = history[-3:]
        avg_delta = (
            sum(recent[i]["val_mAP5095"] - recent[i-1]["val_mAP5095"] for i in range(1, len(recent))) / len(recent)
            if len(recent) >= 2 else 0.0
        )
        history_block = "\n".join(
            f"  exp{i+1:02d}: [{h['status'].upper():7s}] mAP={h['val_mAP5095']:.4f} "
            f"delta={h['val_mAP5095']-(history[i-1]['val_mAP5095'] if i>0 else baseline):+.4f} | {h['description']}"
            for i, h in enumerate(history)
        ) or "  (none yet)"

        num_classes  = PromptBuilder._extract(r"^NC\s*=\s*(\d+)", current_code)
        dataset_size = PromptBuilder._extract(r"^DATASET_SIZE\s*=\s*(.+)", current_code)

        return dict(
            history_block=history_block,
            baseline=baseline, best_map=best_map,
            kept=len(kept), discarded=len(discarded), crashed=len(crashed),
            avg_delta=avg_delta,
            trend="PLATEAU detected" if avg_delta < 0.001 else "Positive trend",
            params_tried="; ".join(h["description"] for h in history[1:]) or "none yet",
            num_classes=num_classes, dataset_size=dataset_size,
            current_code=current_code,
        )

    # ── Scientist prompt (Bùi Đức Toàn) ──────────────────────────────────
    @staticmethod
    def build_scientist(history: list[dict], current_code: str) -> tuple[str, str]:
        c = PromptBuilder._context_block(history, current_code)
        sys_prompt = (
            f"You are {Config.SCIENTIST_NAME}, {Config.SCIENTIST_ROLE}.\n"
            f"You specialize in computer vision and YOLO fine-tuning research.\n\n"
            "Your task: Analyze the experiment history and propose the single most impactful "
            "hyperparameter change. Think like a scientist — form a clear hypothesis.\n\n"
            "OUTPUT FORMAT (strict JSON, no markdown):\n"
            "{\n"
            '  "analysis": "<observation about underfitting/overfitting/plateau>",\n'
            '  "hypothesis": "<if we change X to Y, then Z because ...>",\n'
            '  "proposed_param": "<PARAM_NAME>",\n'
            '  "proposed_value": "<value>",\n'
            '  "expected_delta_map": "<e.g. +0.01 to +0.03>",\n'
            '  "risk": "low|medium|high",\n'
            '  "confidence": <0.0-1.0>,\n'
            '  "reasoning": "<deep scientific rationale>"\n'
            "}"
        )
        usr_prompt = (
            f"=== EXPERIMENT HISTORY ===\n{c['history_block']}\n\n"
            f"=== STATUS ===\n"
            f"Baseline: {c['baseline']:.4f} | Best: {c['best_map']:.4f} | "
            f"Kept/Discard/Crash: {c['kept']}/{c['discarded']}/{c['crashed']}\n"
            f"Recent avg Δ={c['avg_delta']:+.4f} → {c['trend']}\n"
            f"Tried: {c['params_tried']}\n\n"
            f"=== DATASET ===\n{c['num_classes']} classes, ~{c['dataset_size']} images\n\n"
            f"=== CURRENT train.py ===\n{c['current_code']}\n\n"
            f"Fixed (must not change): {', '.join(Config.FIXED_PARAMS)}\n\n"
            "Propose the single best next experiment. Return JSON only."
        )
        return sys_prompt, usr_prompt

    # ── Engineer prompt (Vũ Thế Đệ) ──────────────────────────────────────
    @staticmethod
    def build_engineer(
        history: list[dict],
        current_code: str,
        scientist_proposal: dict,
    ) -> tuple[str, str]:
        c = PromptBuilder._context_block(history, current_code)
        proposal_str = "\n".join(f"  {k}: {v}" for k, v in scientist_proposal.items())
        sys_prompt = (
            f"You are {Config.ENGINEER_NAME}, {Config.ENGINEER_ROLE}.\n"
            f"You receive a research proposal from {Config.SCIENTIST_NAME} ({Config.SCIENTIST_ROLE}) "
            "and your job is to:\n"
            "  1. Review the proposal for engineering feasibility (VRAM budget, valid param range)\n"
            "  2. Accept or refine the proposed change\n"
            "  3. Implement it as a complete, valid train.py\n\n"
            "OUTPUT FORMAT (strict JSON, no markdown):\n"
            "{\n"
            '  "review": "<engineering assessment of the proposal>",\n'
            '  "refinement": "<any adjustments made and why, or \'accepted as-is\'>",\n'
            f'  "description": "REASON: <why> | CHANGE: <what> | EXPECTED: <effect>",\n'
            '  "new_code": "<complete train.py content>"\n'
            "}\n\n"
            "CRITICAL RULES:\n"
            f"  - Do NOT change: {', '.join(Config.FIXED_PARAMS)}\n"
            "  - Change EXACTLY ONE param OR one allowed group (HSV_H/S/V, LR0/LRF, etc.)\n"
            "  - new_code must be valid Python starting with a triple-quoted docstring\n"
            "  - Escape every newline as \\n and every quote as \\\" inside the JSON string"
        )
        usr_prompt = (
            f"=== SCIENTIST PROPOSAL ({Config.SCIENTIST_NAME}) ===\n{proposal_str}\n\n"
            f"=== EXPERIMENT HISTORY ===\n{c['history_block']}\n\n"
            f"=== CURRENT train.py ===\n{c['current_code']}\n\n"
            f"Peak VRAM from history: "
            f"{max((h.get('memory_gb', 0) for h in history), default=0):.1f} GB\n\n"
            "Review the proposal, implement it, return JSON only."
        )
        return sys_prompt, usr_prompt

    # ── Legacy single-agent prompt (unchanged) ────────────────────────────
    @staticmethod
    def build(history: list[dict], current_code: str) -> tuple[str, str]:
        c = PromptBuilder._context_block(history, current_code)
        sys_prompt = (
            "You are a expert computer vision researcher specialising in YOLO fine-tuning.\n"
            f"Goal: maximize val/mAP50-95 on dataset with {c['num_classes']} classes "
            f"(~{c['dataset_size']} images).\n\n"
            "REASONING PROTOCOL:\n"
            "1. Analyze underfitting/overfitting.\n"
            "2. Pick ONE change with highest impact.\n"
            "Encode reasoning in 'description' as: 'REASON: <why> | CHANGE: <what> | EXPECTED: <effect>'\n\n"
            "OUTPUT FORMAT (Strict JSON ONLY):\n"
            '{"description": "...", "new_code": "<complete train.py>"}\n\n'
            "CRITICAL RULES:\n"
            f" - Do NOT change: {', '.join(Config.FIXED_PARAMS)}\n"
            " - Change EXACTLY ONE param OR one allowed group (e.g. HSV_H/S/V, LR0/LRF).\n"
            " - Every newline -> \\n, quote -> \\\", backslash -> \\\\\n"
            " - Must be valid json.loads()\n"
            " - File must start with a triple-quoted docstring"
        )
        usr_prompt = (
            f"=== EXPERIMENT HISTORY ===\n{c['history_block']}\n\n"
            f"=== TREND ===\nBaseline: {c['baseline']:.4f} | Best: {c['best_map']:.4f}\n"
            f"Kept/Discard/Crash: {c['kept']}/{c['discarded']}/{c['crashed']}\n"
            f"Recent avg Δ={c['avg_delta']:+.4f}. {c['trend']}\nTried: {c['params_tried']}\n\n"
            f"=== CURRENT train.py ===\n{c['current_code']}\n\n"
            "Task: Diagnose and select ONE change. Return full modified train.py as JSON."
        )
        return sys_prompt, usr_prompt

    @staticmethod
    def _extract(pattern: str, src: str, default: str = "unknown") -> str:
        m = re.search(pattern, src, re.MULTILINE)
        return m.group(1).strip() if m else default