"""
live_stream_training.py  —  AutoResearch-DET  ·  Beautiful Demo UI
=======================================================================
Layout
  ┌─────────────────────────────────────────────────────────┐
  │  Header + Config row + Start / Stop                     │
  ├─────────────────────────────────────────────────────────┤
  │  💬  Agent Discussion  (chatbot, scrollable)            │
  │      Bùi Đức Toàn  ↔  Vũ Thế Đệ                       │
  ├─────────────────────────────────────────────────────────┤
  │  🏋️  Training Progress  (appears once training starts)  │
  │      Progress bar  ·  Current metrics  ·  Exp table     │
  └─────────────────────────────────────────────────────────┘

Fixes vs original
  • ❌ signal only works in main thread  →  guarded in pipeline.py (see PATCH)
  • Gradio chatbot replaces raw Textbox panels
  • Training section with animated progress bar + live metrics
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from core.pipeline import AutoResearchPipeline
from core.history import HistoryManager

# ── Global state ──────────────────────────────────────────────────────────────

_event_q: queue.Queue = queue.Queue()
_pipeline_thread: threading.Thread | None = None
_running = threading.Event()


def _event_cb(evt: dict):
    _event_q.put(evt)


# ── Pipeline runner (background thread) ───────────────────────────────────────

def _run_pipeline(experiments, scientist_model, engineer_model, gpus, dry_run, quiet):
    _running.set()
    try:
        args = SimpleNamespace(
            experiments=experiments,
            scientist_model=scientist_model,
            engineer_model=engineer_model,
            model=None,
            gpus=gpus,
            cuda_devices=None,
            resume=False,
            dry_run=dry_run,
            quiet=quiet,
        )
        pipeline = AutoResearchPipeline(args, event_cb=_event_cb)
        pipeline.run()
    except Exception as exc:
        _event_q.put({"agent": "pipeline", "kind": "error", "content": str(exc)})
    finally:
        _event_q.put({"agent": "pipeline", "kind": "done", "content": "🏁 Pipeline finished."})
        _running.clear()


# ── Display state ─────────────────────────────────────────────────────────────

class DisplayState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.chat_messages: list[dict] = []   # [{role, content}]
        self.training_log  = ""
        self.history_rows  = []
        self.status        = "Idle"
        self.exp_done      = 0
        self.exp_total     = 10
        self.last_metrics: dict = {}
        self._sci_buffer   = {}   # accumulate scientist fields before posting
        self._training_active = False


_state = DisplayState()


def _post_chat(role: str, content: str):
    """Append a message to the chat history."""
    _state.chat_messages.append({"role": role, "content": content})


def _process_events():
    events = []
    while True:
        try:
            events.append(_event_q.get_nowait())
        except queue.Empty:
            break

    sci = Config.SCIENTIST_NAME
    eng = Config.ENGINEER_NAME

    for evt in events:
        agent   = evt.get("agent", "")
        kind    = evt.get("kind", "")
        content = evt.get("content", "")

        # ── Scientist events ──────────────────────────────────────────────
        if agent == sci:
            if kind == "status":
                _state.status = f"[{sci}] {content}"
                _post_chat("user", f"**🔬 {sci}** — *{content}*")

            elif kind == "parsed":
                try:
                    obj = json.loads(content)
                    analysis   = obj.get("analysis", "")
                    hypothesis = obj.get("hypothesis", "")
                    confidence = obj.get("confidence", "")
                    reasoning  = obj.get("reasoning", "")
                    param      = obj.get("proposed_param", "")
                    value      = obj.get("proposed_value", "")

                    msg_lines = [f"**🧪 {sci}** *(Ph.D AI Research Scientist)*\n"]
                    if analysis:
                        msg_lines.append(f"📊 **Analysis:**\n{analysis}\n")
                    if hypothesis:
                        msg_lines.append(f"💡 **Hypothesis:**\n{hypothesis}\n")
                    if param:
                        msg_lines.append(f"🎯 **Proposed:** `{param}` → `{value}`")
                    if confidence:
                        pct = int(float(confidence) * 100) if isinstance(confidence, (int, float)) else confidence
                        msg_lines.append(f"📈 **Confidence:** {pct}%")
                    if reasoning:
                        msg_lines.append(f"\n🧠 **Reasoning:**\n{reasoning}")

                    _post_chat("user", "\n".join(msg_lines))
                except Exception:
                    _post_chat("user", f"**🧪 {sci}**\n{content[:800]}")

        # ── Engineer events ───────────────────────────────────────────────
        elif agent == eng:
            if kind == "status":
                _state.status = f"[{eng}] {content}"
                _post_chat("assistant", f"**⚙️ {eng}** — *{content}*")

            elif kind == "review":
                lines = content.split("\n")
                review    = lines[0].replace("Review: ", "") if lines else ""
                refinement = lines[1].replace("Refinement: ", "") if len(lines) > 1 else ""
                msg = [f"**⚙️ {eng}** *(Lead AI Engineer)*\n"]
                if review:
                    msg.append(f"🔍 **Review:**\n{review}\n")
                if refinement:
                    msg.append(f"🔧 **Refinement:**\n{refinement}")
                _post_chat("assistant", "\n".join(msg))

            elif kind == "response":
                try:
                    i = content.find("{")
                    if i >= 0:
                        obj, _ = json.JSONDecoder().raw_decode(content, i)
                        desc = obj.get("description", "")
                        if desc:
                            _post_chat("assistant",
                                       f"**⚙️ {eng}**\n✅ **Implementation ready:** {desc}")
                except Exception:
                    pass

        # ── Pipeline events ───────────────────────────────────────────────
        elif agent == "pipeline":
            if kind == "training_start":
                _state.status = f"🏋️ Training: {content}"
                _state.training_log += f"\n▶ {content}\n"
                _state.exp_done = 0
                _state._training_active = True
                _post_chat("assistant",
                           f"**🏋️ Training started**\n```\n{content}\n```")

            elif kind == "exp_result":
                status   = evt.get("status", "")
                exp_num  = evt.get("exp_num", 0)
                metrics  = evt.get("metrics", {})
                desc     = evt.get("description", "")
                proposal = evt.get("scientist_proposal") or {}

                _state.exp_done = exp_num
                _state.last_metrics = metrics

                icon = "✅" if status == "keep" else ("❌" if status == "discard" else "💥")
                map5095 = metrics.get("val_mAP5095", 0)
                map50   = metrics.get("val_mAP50",   0)
                vram    = metrics.get("peak_vram_mb", 0) / 1024

                _state.training_log += (
                    f"{icon} exp{exp_num:02d}  {status.upper()}  "
                    f"mAP={map5095:.4f}  VRAM={vram:.1f}GB\n"
                )
                _state.history_rows.append([
                    f"exp{exp_num:02d}", desc,
                    f"{map5095:.4f}", f"{map50:.4f}",
                    f"{vram:.1f}", status.upper(),
                    proposal.get("proposed_param", ""),
                    f"{proposal.get('confidence', '')}",
                ])
                _state.status = (
                    f"{'✅ KEEP' if status=='keep' else '❌ DISCARD'} "
                    f"exp{exp_num:02d}  mAP={map5095:.4f}"
                )
                _post_chat(
                    "user",
                    f"**📊 Experiment {exp_num:02d} Result**\n"
                    f"{icon} **{status.upper()}**  ·  mAP50-95: `{map5095:.4f}`  ·  mAP50: `{map50:.4f}`  ·  VRAM: `{vram:.1f} GB`\n"
                    f"*{desc}*"
                )

            elif kind in ("error", "done"):
                icon = "❌" if kind == "error" else "🏁"
                _state.status = content
                _state.training_log += f"\n{icon} {content}\n"
                _post_chat("assistant", f"**{icon} {content}**")
                _state._training_active = False

    return events


# ── Progress bar HTML ─────────────────────────────────────────────────────────

def _progress_html(done: int, total: int, metrics: dict) -> str:
    pct = int(done / max(total, 1) * 100)
    map_val  = metrics.get("val_mAP5095", None)
    map50    = metrics.get("val_mAP50",   None)
    vram     = metrics.get("peak_vram_mb", None)

    metric_html = ""
    if map_val is not None:
        metric_html = f"""
        <div style="display:flex;gap:24px;margin-top:12px;flex-wrap:wrap;">
          <div class="metric-chip">📊 mAP50-95 <span class="metric-val">{map_val:.4f}</span></div>
          <div class="metric-chip">🎯 mAP50 <span class="metric-val">{map50:.4f}</span></div>
          <div class="metric-chip">💾 VRAM <span class="metric-val">{vram/1024:.1f} GB</span></div>
        </div>"""

    return f"""
    <div class="prog-wrap">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
        <span style="font-weight:600;font-size:0.9em;">Experiments: {done} / {total}</span>
        <span style="font-weight:700;color:#7dd3fc;">{pct}%</span>
      </div>
      <div class="prog-track">
        <div class="prog-bar" style="width:{pct}%;"></div>
      </div>
      {metric_html}
    </div>"""


# ── Tick function ─────────────────────────────────────────────────────────────

def _tick():
    _process_events()

    rows = _state.history_rows or [["—"] * 8]
    pbar = _progress_html(_state.exp_done, _state.exp_total, _state.last_metrics)

    chat = [[m["content"], None] if m["role"] == "user" else [None, m["content"]]
            for m in _state.chat_messages]

    return (
        chat,
        pbar,
        _state.training_log[-8000:],
        rows,
        _state.status,
        gr.update(interactive=not _running.is_set()),
        gr.update(interactive=_running.is_set()),
    )


# ── Start / Stop ──────────────────────────────────────────────────────────────

def start_pipeline(experiments, scientist_model, engineer_model, gpus, dry_run, quiet):
    global _pipeline_thread
    if _running.is_set():
        return "⚠️ Already running."
    _state.reset()
    _state.exp_total = int(experiments)
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
        _event_q.put({"agent": "pipeline", "kind": "done", "content": "⏹ Stop requested."})
        _running.clear()
        return "⏹ Stop signal sent."
    return "Not running."


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
/* ── Base ── */
body, .gradio-container {
    background: #0f172a !important;
    color: #e2e8f0 !important;
}

/* ── Header ── */
.ar-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2d4a 60%, #162032 100%);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 4px;
    border: 1px solid #1e40af44;
}
.ar-header h1 {
    margin: 0 0 4px 0;
    font-size: 1.7em;
    font-weight: 800;
    background: linear-gradient(90deg, #7dd3fc, #a5f3fc, #bfdbfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.ar-header p {
    margin: 0;
    color: #94a3b8;
    font-size: 0.9em;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.8em;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin: 16px 0 6px 0;
    padding-left: 4px;
}

/* ── Chatbot ── */
#agent-chat .message-wrap { gap: 12px; }
#agent-chat { background: #111827 !important; border-radius: 12px !important; border: 1px solid #1e293b !important; }
/* user bubble = Scientist (blue left accent) */
#agent-chat .user { background: #1e3a5f !important; border-left: 3px solid #3b82f6 !important; }
/* bot bubble = Engineer (green left accent) */
#agent-chat .bot  { background: #14291e !important; border-left: 3px solid #22c55e !important; }

/* ── Config panel ── */
.config-row label { font-size: 0.82em !important; color: #94a3b8 !important; }

/* ── Status bar ── */
#status-bar textarea {
    background: #1e293b !important;
    border: none !important;
    border-radius: 8px !important;
    color: #7dd3fc !important;
    font-size: 0.85em !important;
    padding: 6px 12px !important;
}

/* ── Buttons ── */
#btn-start { background: linear-gradient(135deg,#2563eb,#1d4ed8) !important; border:none !important; font-weight:700 !important; }
#btn-stop  { background: linear-gradient(135deg,#dc2626,#b91c1c) !important; border:none !important; font-weight:700 !important; }

/* ── Progress bar ── */
.prog-wrap { background:#1e293b; border-radius:10px; padding:16px 20px; border:1px solid #334155; }
.prog-track { background:#0f172a; border-radius:8px; height:14px; overflow:hidden; }
.prog-bar   {
    height:100%; border-radius:8px;
    background: linear-gradient(90deg, #2563eb, #06b6d4, #7dd3fc);
    transition: width 0.6s ease;
    box-shadow: 0 0 12px #3b82f688;
}
.metric-chip {
    background:#0f172a; border:1px solid #1e3a5f;
    border-radius:8px; padding:6px 14px;
    font-size:0.82em; color:#94a3b8; display:flex; gap:8px; align-items:center;
}
.metric-val { color:#7dd3fc; font-weight:700; }

/* ── Log ── */
#live-log textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.8em !important;
    background: #0a0f1a !important;
    color: #94a3b8 !important;
    border-color: #1e293b !important;
}

/* ── Table ── */
.gr-dataframe { background: #111827 !important; border-radius: 10px !important; }
.gr-dataframe thead th { background: #1e293b !important; color: #7dd3fc !important; font-weight:700 !important; }
.gr-dataframe tbody tr:hover { background: #1e3a5f22 !important; }
"""


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui():
    sci = Config.SCIENTIST_NAME
    eng = Config.ENGINEER_NAME

    with gr.Blocks(css=CSS, title="AutoResearch-DET", theme=gr.themes.Base()) as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML(f"""
        <div class="ar-header">
          <h1>🔬 AutoResearch-DET</h1>
          <p>Multi-Agent YOLO Hyperparameter Search &nbsp;·&nbsp;
             <strong style="color:#7dd3fc">{sci}</strong> (Scientist)
             &nbsp;↔&nbsp;
             <strong style="color:#4ade80">{eng}</strong> (Engineer)</p>
        </div>
        """)

        # ── Config row ────────────────────────────────────────────────────
        gr.HTML('<div class="section-label">⚙️ Configuration</div>')
        with gr.Row(elem_classes=["config-row"]):
            i_experiments     = gr.Slider(1, 20, value=10, step=1, label="Experiments")
            i_scientist_model = gr.Textbox(Config.SCIENTIST_MODEL, label=f"Model — {sci}")
            i_engineer_model  = gr.Textbox(Config.ENGINEER_MODEL,  label=f"Model — {eng}")
            i_gpus            = gr.Slider(0, 8, value=1, step=1, label="GPUs")

        with gr.Row():
            i_dry_run  = gr.Checkbox(False, label="Dry-run (no training)")
            i_quiet    = gr.Checkbox(True,  label="Quiet training log")
            btn_start  = gr.Button("▶  Start", variant="primary", elem_id="btn-start")
            btn_stop   = gr.Button("⏹  Stop",  variant="stop",    elem_id="btn-stop")

        status_bar = gr.Textbox("Idle", show_label=False, elem_id="status-bar",
                                interactive=False, max_lines=1)

        # ── Agent Discussion (chatbot) ─────────────────────────────────────
        gr.HTML('<div class="section-label">💬 Agent Discussion</div>')
        chatbot = gr.Chatbot(
            value=[],
            elem_id="agent-chat",
            height=420,
            show_label=False,
            bubble_full_width=False,
            avatar_images=(
                "https://api.dicebear.com/8.x/bottts/svg?seed=scientist&backgroundColor=1e3a5f",
                "https://api.dicebear.com/8.x/bottts/svg?seed=engineer&backgroundColor=14291e",
            ),
        )

        # ── Training Progress ──────────────────────────────────────────────
        gr.HTML('<div class="section-label">🏋️ Training Progress</div>')
        progress_html = gr.HTML(_progress_html(0, 10, {}))

        with gr.Accordion("📄 Live Training Log", open=False):
            o_log = gr.Textbox(lines=12, max_lines=12, interactive=False,
                               show_label=False, elem_id="live-log")

        gr.HTML('<div class="section-label">📊 Experiment History</div>')
        o_table = gr.Dataframe(
            headers=["Exp", "Description", "mAP50-95", "mAP50",
                     "VRAM(GB)", "Status", "Param", "Confidence"],
            datatype=["str"] * 8,
            value=[["—"] * 8],
            interactive=False,
            wrap=True,
        )

        # ── Timer ─────────────────────────────────────────────────────────
        timer = gr.Timer(1.5)
        outputs = [chatbot, progress_html, o_log, o_table, status_bar, btn_start, btn_stop]
        timer.tick(_tick, outputs=outputs)

        # ── Handlers ──────────────────────────────────────────────────────
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