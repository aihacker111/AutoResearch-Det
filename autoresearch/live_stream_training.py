"""
app_gradio.py — Live demo UI for AutoResearch-DET multi-agent pipeline.

Shows real-time reasoning from both agents:
  • Bùi Đức Toàn  (Ph.D AI Research Scientist)
  • Vũ Thế Đệ     (Lead AI Engineer)

Run:
  export OPENROUTER_API_KEY=...
  python app_gradio.py
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from types   import SimpleNamespace

import gradio as gr

# Add parent to path so imports work when run from the project root
import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config         import Config
from core.pipeline   import AutoResearchPipeline
from core.history    import HistoryManager

# ── Global state ─────────────────────────────────────────────────────────────

_event_q: queue.Queue = queue.Queue()
_pipeline_thread: threading.Thread | None = None
_running = threading.Event()


# ── Event callback → queue ────────────────────────────────────────────────────

def _event_cb(evt: dict):
    _event_q.put(evt)


# ── Pipeline runner (background thread) ──────────────────────────────────────

def _run_pipeline(experiments: int, scientist_model: str, engineer_model: str,
                  gpus: int, dry_run: bool, quiet: bool):
    _running.set()
    try:
        args = SimpleNamespace(
            experiments     = experiments,
            scientist_model = scientist_model,
            engineer_model  = engineer_model,
            model           = None,      # use multi-agent
            gpus            = gpus,
            cuda_devices    = None,
            resume          = False,
            dry_run         = dry_run,
            quiet           = quiet,
        )
        pipeline = AutoResearchPipeline(args, event_cb=_event_cb)
        pipeline.run()
    except Exception as exc:
        _event_q.put({"agent": "pipeline", "kind": "error", "content": str(exc)})
    finally:
        _event_q.put({"agent": "pipeline", "kind": "done", "content": "Pipeline finished."})
        _running.clear()


# ── Drain event queue into display state ─────────────────────────────────────

class DisplayState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sci_analysis    = ""
        self.sci_hypothesis  = ""
        self.sci_confidence  = ""
        self.sci_reasoning   = ""
        self.eng_review      = ""
        self.eng_refinement  = ""
        self.eng_description = ""
        self.training_log    = ""
        self.history_rows    = []
        self.status          = "Idle"


_state = DisplayState()


def _process_events():
    """Drain the queue and update DisplayState. Returns list of new events."""
    events = []
    while True:
        try:
            evt = _event_q.get_nowait()
            events.append(evt)
        except queue.Empty:
            break

    for evt in events:
        agent   = evt.get("agent", "")
        kind    = evt.get("kind",  "")
        content = evt.get("content", "")

        sci_name = Config.SCIENTIST_NAME
        eng_name = Config.ENGINEER_NAME

        if agent == sci_name:
            if kind == "status":
                _state.status = f"[{sci_name}] {content}"
            elif kind == "parsed":
                try:
                    obj = json.loads(content)
                    _state.sci_analysis   = obj.get("analysis",   "")
                    _state.sci_hypothesis = obj.get("hypothesis",  "")
                    _state.sci_confidence = str(obj.get("confidence", ""))
                    _state.sci_reasoning  = obj.get("reasoning",   "")
                except Exception:
                    _state.sci_reasoning = content

        elif agent == eng_name:
            if kind == "status":
                _state.status = f"[{eng_name}] {content}"
            elif kind == "review":
                lines = content.split("\n")
                _state.eng_review      = lines[0].replace("Review: ", "")
                _state.eng_refinement  = lines[1].replace("Refinement: ", "") if len(lines) > 1 else ""
            elif kind == "response":
                # Try to extract description from engineer JSON
                try:
                    i = content.find("{")
                    if i >= 0:
                        obj, _ = json.JSONDecoder().raw_decode(content, i)
                        _state.eng_description = obj.get("description", "")
                        if not _state.eng_review:
                            _state.eng_review     = obj.get("review", "")
                            _state.eng_refinement = obj.get("refinement", "")
                except Exception:
                    pass

        elif agent == "pipeline":
            if kind == "training_start":
                _state.status = f"🏋️ Training: {content}"
                _state.training_log += f"\n▶ {content}\n"
            elif kind == "exp_result":
                status   = evt.get("status", "")
                exp_num  = evt.get("exp_num", "")
                metrics  = evt.get("metrics", {})
                desc     = evt.get("description", "")
                proposal = evt.get("scientist_proposal") or {}
                icon     = "✔" if status == "keep" else ("✗" if status == "discard" else "💥")
                _state.training_log += (
                    f"{icon} exp{exp_num:02d}  {status.upper()}  "
                    f"mAP={metrics.get('val_mAP5095', 0):.4f}  "
                    f"VRAM={metrics.get('peak_vram_mb', 0)/1024:.1f}GB\n"
                )
                _state.history_rows.append([
                    f"exp{exp_num:02d}", desc,
                    f"{metrics.get('val_mAP5095',0):.4f}",
                    f"{metrics.get('val_mAP50',0):.4f}",
                    f"{metrics.get('peak_vram_mb',0)/1024:.1f}",
                    status.upper(),
                    proposal.get("proposed_param", ""),
                    f"{proposal.get('confidence', '')}",
                ])
                _state.status = f"{'✔ KEEP' if status=='keep' else '✗ DISCARD'} exp{exp_num:02d}  mAP={metrics.get('val_mAP5095',0):.4f}"
            elif kind in ("error", "done"):
                _state.status = content
                _state.training_log += f"\n{'❌' if kind=='error' else '🏁'} {content}\n"

    return events


# ── Gradio polling tick ───────────────────────────────────────────────────────

def _tick():
    """Called by gr.Timer; returns all display component updates."""
    _process_events()

    rows = _state.history_rows or [["—", "—", "—", "—", "—", "—", "—", "—"]]
    return (
        # Scientist panel
        _state.sci_analysis,
        _state.sci_hypothesis,
        _state.sci_confidence,
        _state.sci_reasoning,
        # Engineer panel
        _state.eng_review,
        _state.eng_refinement,
        _state.eng_description,
        # Bottom
        _state.training_log[-6000:],     # keep last 6k chars
        rows,
        _state.status,
        gr.update(interactive=not _running.is_set()),   # start button
        gr.update(interactive=_running.is_set()),        # stop button
    )


# ── Start / Stop handlers ─────────────────────────────────────────────────────

def start_pipeline(experiments, scientist_model, engineer_model, gpus, dry_run, quiet):
    global _pipeline_thread
    if _running.is_set():
        return "Already running."
    _state.reset()
    while not _event_q.empty():
        try: _event_q.get_nowait()
        except queue.Empty: break

    _pipeline_thread = threading.Thread(
        target=_run_pipeline,
        args=(int(experiments), scientist_model, engineer_model,
              int(gpus), dry_run, quiet),
        daemon=True,
    )
    _pipeline_thread.start()
    return "🚀 Pipeline started…"


def stop_pipeline():
    if _pipeline_thread and _pipeline_thread.is_alive():
        _event_q.put({"agent": "pipeline", "kind": "done", "content": "Stop requested."})
        # The pipeline checks shutdown_requested via SIGINT; we set a flag via the event
        _running.clear()
        return "⏹ Stop signal sent."
    return "Not running."


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
.agent-card   { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 12px; }
.sci-card     { border-left: 4px solid #4f8ef7; }
.eng-card     { border-left: 4px solid #22c55e; }
.agent-header { font-size: 1.1em; font-weight: 700; margin-bottom: 4px; }
.agent-role   { font-size: 0.82em; color: var(--body-text-color-subdued); margin-bottom: 10px; }
#status-bar   { font-size: 0.9em; padding: 6px 10px; background: var(--background-fill-secondary);
                border-radius: 6px; min-height: 32px; }
"""

def build_ui():
    sci = Config.SCIENTIST_NAME
    eng = Config.ENGINEER_NAME
    sci_role = Config.SCIENTIST_ROLE
    eng_role = Config.ENGINEER_ROLE

    with gr.Blocks(css=CSS, title="AutoResearch-DET") as demo:

        gr.Markdown(
            f"# 🔬 AutoResearch-DET  —  Multi-Agent YOLO Hyperparameter Search\n"
            f"**{sci}** ({sci_role}) · **{eng}** ({eng_role})"
        )

        # ── Config row ────────────────────────────────────────────────────
        with gr.Row():
            i_experiments    = gr.Slider(1, 20, value=10, step=1, label="Experiments")
            i_scientist_model= gr.Textbox(Config.SCIENTIST_MODEL, label=f"Model — {sci}")
            i_engineer_model = gr.Textbox(Config.ENGINEER_MODEL,  label=f"Model — {eng}")
            i_gpus           = gr.Slider(0, 8, value=1, step=1, label="GPUs")

        with gr.Row():
            i_dry_run = gr.Checkbox(False, label="Dry-run (no training)")
            i_quiet   = gr.Checkbox(True,  label="Quiet training log")
            btn_start = gr.Button("▶  Start",  variant="primary")
            btn_stop  = gr.Button("⏹  Stop",   variant="stop")

        status_bar = gr.Textbox("Idle", show_label=False, elem_id="status-bar",
                                interactive=False, max_lines=1)

        # ── Agent panels ──────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(elem_classes=["agent-card", "sci-card"]):
                gr.HTML(f'<div class="agent-header">🧪 {sci}</div>'
                        f'<div class="agent-role">{sci_role}</div>')
                o_sci_analysis   = gr.Textbox(label="Analysis",   lines=3, interactive=False)
                o_sci_hypothesis = gr.Textbox(label="Hypothesis",  lines=3, interactive=False)
                o_sci_confidence = gr.Textbox(label="Confidence",  lines=1, interactive=False)
                o_sci_reasoning  = gr.Textbox(label="Deep Reasoning", lines=6, interactive=False)

            with gr.Column(elem_classes=["agent-card", "eng-card"]):
                gr.HTML(f'<div class="agent-header">⚙️ {eng}</div>'
                        f'<div class="agent-role">{eng_role}</div>')
                o_eng_review      = gr.Textbox(label="Engineering Review", lines=3, interactive=False)
                o_eng_refinement  = gr.Textbox(label="Refinement",          lines=3, interactive=False)
                o_eng_description = gr.Textbox(label="Final Description",   lines=2, interactive=False)

        # ── Results table ─────────────────────────────────────────────────
        gr.Markdown("### 📊 Experiment History")
        o_table = gr.Dataframe(
            headers=["Exp", "Description", "mAP50-95", "mAP50",
                     "VRAM(GB)", "Status", "Param", "Confidence"],
            datatype=["str"] * 8,
            value=[["—"] * 8],
            interactive=False,
            wrap=True,
        )

        # ── Training log ──────────────────────────────────────────────────
        gr.Markdown("### 📄 Live Log")
        o_log = gr.Textbox(lines=14, max_lines=14, interactive=False,
                           show_label=False, elem_id="live-log")

        # ── Timer (polls every 1.5 s) ─────────────────────────────────────
        timer = gr.Timer(1.5)
        outputs = [
            o_sci_analysis, o_sci_hypothesis, o_sci_confidence, o_sci_reasoning,
            o_eng_review, o_eng_refinement, o_eng_description,
            o_log, o_table, status_bar, btn_start, btn_stop,
        ]
        timer.tick(_tick, outputs=outputs)

        # ── Button handlers ───────────────────────────────────────────────
        btn_start.click(
            start_pipeline,
            inputs=[i_experiments, i_scientist_model, i_engineer_model,
                    i_gpus, i_dry_run, i_quiet],
            outputs=[status_bar],
        )
        btn_stop.click(stop_pipeline, outputs=[status_bar])

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not Config.OPENROUTER_API_KEY:
        print("WARNING: OPENROUTER_API_KEY not set — LLM calls will fail.")
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)