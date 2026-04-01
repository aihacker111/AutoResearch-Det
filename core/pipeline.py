"""
pipeline.py — Multi-agent AutoResearch pipeline.

Experiment loop now uses a two-agent system:
  • Bùi Đức Toàn (Scientist) proposes the next hyperparameter change.
  • Vũ Thế Đệ   (Engineer) reviews and implements the final train.py.

An optional `event_cb(dict)` lets callers (e.g. Gradio app) stream
agent reasoning to the UI in real time.
"""

from __future__ import annotations

import signal
import sys
import threading
from typing import Callable

from config                import Config
from core.history           import HistoryManager
from agent.multi_agent      import MultiAgentCoordinator
from agent.prompt_builder   import PromptBuilder
from agent.llm_client       import LLMClient
from agent.parser            import LLMParser
from executor.git_ops        import GitManager
from executor.trainer        import Trainer
from executor.validator      import Validator
from utils.metrics           import MetricsParser
from utils.plotter           import Plotter


class AutoResearchPipeline:

    def __init__(self, args, event_cb: Callable[[dict], None] | None = None):
        self.args     = args
        self.event_cb = event_cb or (lambda e: None)
        self.history  = HistoryManager()
        self.trainer  = Trainer(
            num_gpus=args.gpus,
            cuda_devices=getattr(args, "cuda_devices", None),
            quiet=getattr(args, "quiet", False),
        )
        self.baseline_time      = None
        self.shutdown_requested = False

        # ── Choose single-agent or multi-agent ────────────────────────────
        scientist_model = getattr(args, "scientist_model", Config.SCIENTIST_MODEL)
        engineer_model  = getattr(args, "engineer_model",  Config.ENGINEER_MODEL)

        if getattr(args, "model", None):
            # Legacy --model flag → single-agent fallback
            self._multi_agent = False
            self.llm = LLMClient(model=args.model)
        else:
            self._multi_agent = True
            self.coordinator  = MultiAgentCoordinator(
                scientist_model=scientist_model,
                engineer_model=engineer_model,
                event_cb=self.event_cb,
            )

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._handle_sigint)

    # ── Signal handling ───────────────────────────────────────────────────

    def _handle_sigint(self, sig, frame):
        print("\n[!] Interrupt received — finishing current experiment then stopping.")
        self.shutdown_requested = True

    # ── Proposal (wraps both single/multi agent) ──────────────────────────

    def _get_proposal(self, current_code: str) -> dict:
        if self._multi_agent:
            return self.coordinator.propose(self.history.records, current_code)
        # Legacy single-agent path
        sys_p, usr_p = PromptBuilder.build(self.history.records, current_code)
        raw          = self.llm.generate(sys_p, usr_p)
        result       = LLMParser.parse(raw)
        self.event_cb({"agent": "legacy", "kind": "response", "content": raw})
        return result

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self):
        branch = GitManager.setup_branch()
        if not Config.TRAIN_FILE.is_file():
            sys.exit(f"ERROR: {Config.TRAIN_FILE} not found.")

        if getattr(self.args, "resume", False):
            self.history.load()

        completed = self.history.get_completed_count()
        best_map  = self.history.get_best_map()

        mode_str = "multi-agent" if self._multi_agent else "single-agent"
        sci = Config.SCIENTIST_NAME
        eng = Config.ENGINEER_NAME

        print(f"\n{'='*62}")
        print(f"  AutoResearch-DET  |  YOLO11 fine-tuning  [{mode_str}]")
        print(f"  Branch       : {branch}")
        if self._multi_agent:
            print(f"  Scientist    : {sci} ({Config.SCIENTIST_MODEL})")
            print(f"  Engineer     : {eng} ({Config.ENGINEER_MODEL})")
        else:
            print(f"  LLM model    : {self.args.model}")
        print(f"  GPUs         : {self.trainer.num_gpus}")
        print(f"  Experiments  : {self.args.experiments - completed} remaining / {self.args.experiments} total")
        print(f"  Fixed params : {', '.join(Config.FIXED_PARAMS)}")
        if getattr(self.args, "dry_run", False):
            print("  Mode         : DRY-RUN (no training)")
        print(f"{'='*62}\n")

        for exp_num in range(completed + 1, self.args.experiments + 1):
            if self.shutdown_requested:
                print("\n[!] Graceful shutdown — results saved.")
                break

            self.event_cb({"agent": "pipeline", "kind": "exp_start", "content": str(exp_num)})
            print(f"\n{'='*62}")
            print(f"  Experiment {exp_num:02d} / {self.args.experiments:02d}")
            print(f"{'='*62}")

            current_code = Config.TRAIN_FILE.read_text(errors="ignore")

            if exp_num == 1:
                description = "baseline — default hyperparameters"
                print(f"  Idea  : {description}")
                proposal    = None
            else:
                print(f"  [{sci}] proposing hypothesis...")
                print(f"  [{eng}]  reviewing & implementing...")
                try:
                    result      = self._get_proposal(current_code)
                    description = result["description"]
                    proposal    = result.get("scientist_proposal")

                    if not getattr(self.args, "dry_run", False):
                        clean = Validator.sanitize(result["new_code"])
                        Validator.validate_syntax(clean)
                        Validator.validate_fixed_params(clean, current_code)
                        Validator.validate_single_change(clean, current_code)
                        Config.TRAIN_FILE.write_text(clean, encoding="utf-8")
                    print(f"  Idea  : {description}")
                except Exception as exc:
                    print(f"  Agent error: {exc} — skipping")
                    self.event_cb({"agent": "pipeline", "kind": "error", "content": str(exc)})
                    continue

            if getattr(self.args, "dry_run", False):
                print("  [dry-run] skipping training")
                self.history.append(
                    "dryrun",
                    {"val_mAP5095": 0.0, "val_mAP50": 0.0, "peak_vram_mb": 0.0},
                    "dry-run", description,
                )
                continue

            commit  = GitManager.commit(f"experiment: {description}")
            timeout = self.baseline_time * Config.TIMEOUT_FACTOR if self.baseline_time else None
            print(f"  Commit: {commit}  |  timeout: {f'{timeout:.0f}s' if timeout else 'none'}")
            print("  Training...\n")
            sys.stdout.flush()

            self.event_cb({"agent": "pipeline", "kind": "training_start",
                           "content": f"exp{exp_num:02d}: {description}"})
            success, elapsed = self.trainer.run(timeout)

            if exp_num == 1:
                self.baseline_time = elapsed

            metrics, metrics_ok = MetricsParser.parse()
            val_map = metrics["val_mAP5095"]

            print()
            if not metrics_ok:
                print(f"  ✗  CRASH  ({elapsed:.0f}s)  — see {Config.LOG_FILE}")
                self.history.append(commit, metrics, "crash", description)
                GitManager.rollback()
                self.event_cb({"agent": "pipeline", "kind": "exp_result",
                               "content": f"CRASH exp{exp_num:02d}"})
                continue

            if not success:
                print("  [!] Training exited non-zero but metrics recovered.")

            prev   = self.history.records[-1]["val_mAP5095"] if self.history.records else 0.0
            delta  = val_map - prev
            print(f"  mAP50-95={val_map:.4f}  mAP50={metrics['val_mAP50']:.4f}  "
                  f"VRAM={metrics['peak_vram_mb']/1024:.1f}GB  time={elapsed:.0f}s")

            if val_map > best_map:
                best_map = val_map
                status   = "keep"
                print(f"  ✔  KEEP   (delta={delta:+.4f})")
            else:
                status = "discard"
                print(f"  ✗  DISCARD (delta={delta:+.4f})")
                if exp_num > 1:
                    GitManager.rollback()

            self.history.append(commit, metrics, status, description)
            Plotter.update_plot()

            self.event_cb({
                "agent":   "pipeline",
                "kind":    "exp_result",
                "content": f"exp{exp_num:02d}  {status.upper()}  mAP={val_map:.4f}  Δ={delta:+.4f}  {description}",
                "metrics": metrics,
                "status":  status,
                "exp_num": exp_num,
                "description": description,
                "scientist_proposal": proposal,
            })