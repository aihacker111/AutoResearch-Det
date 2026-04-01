import re
from config import Config

class PromptBuilder:
    @staticmethod
    def build(history: list[dict], current_code: str) -> tuple[str, str]:
        kept = [h for h in history if h["status"] == "keep"]
        crashed = [h for h in history if h["status"] == "crash"]
        discarded = [h for h in history if h["status"] == "discard"]
        
        baseline = history[0]["val_mAP5095"] if history else 0.0
        best_map = max((h["val_mAP5095"] for h in history), default=0.0)
        
        recent = history[-3:]
        avg_delta = sum(recent[i]["val_mAP5095"] - recent[i-1]["val_mAP5095"] for i in range(1, len(recent))) / len(recent) if len(recent) >= 2 else 0.0

        history_block = "\n".join(
            f"  exp{i+1:02d}: [{h['status'].upper():7s}] mAP={h['val_mAP5095']:.4f}  "
            f"delta={h['val_mAP5095'] - (history[i-1]['val_mAP5095'] if i > 0 else baseline):+.4f} | {h['description']}"
            for i, h in enumerate(history)
        ) or "  (none yet)"

        trend_note = f"Recent avg Δ={avg_delta:+.4f}. " + ("PLATEAU detected" if avg_delta < 0.001 else "Positive trend")
        params_tried = "; ".join([h["description"] for h in history[1:]]) or "none yet"

        num_classes = PromptBuilder._extract(r"^NC\s*=\s*(\d+)", current_code)
        dataset_size = PromptBuilder._extract(r"^DATASET_SIZE\s*=\s*(.+)", current_code)

        sys_prompt = (
            "You are a expert computer vision researcher specialising in YOLO fine-tuning.\n"
            f"Goal: maximize val/mAP50-95 on dataset with {num_classes} classes (~{dataset_size} images).\n\n"
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
            f"=== EXPERIMENT HISTORY ===\n{history_block}\n\n"
            f"=== TREND ===\nBaseline: {baseline:.4f} | Best: {best_map:.4f}\nKept/Discard/Crash: {len(kept)}/{len(discarded)}/{len(crashed)}\n{trend_note}\nTried: {params_tried}\n\n"
            f"=== CURRENT train.py ===\n{current_code}\n\n"
            "Task: Diagnose and select ONE change. Return full modified train.py as JSON."
        )

        return sys_prompt, usr_prompt

    @staticmethod
    def _extract(pattern: str, src: str, default="unknown") -> str:
        m = re.search(pattern, src, re.MULTILINE)
        return m.group(1).strip() if m else default