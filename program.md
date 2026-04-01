# autoresearch-det — Multi-Agent YOLO11 Fine-tuning

Two AI agents collaborate to find the best fine-tuning configuration for YOLO11
on a custom detection dataset, maximising `val/mAP50-95` (primary) and
`val/mAP50` (secondary).

---

## Agents

| Role | Name | Model | Responsibility |
|---|---|---|---|
| Ph.D AI Research Scientist | **Bùi Đức Toàn** | `SCIENTIST_MODEL` (e.g. claude-3.5-sonnet) | Analyze history → form hypothesis → propose param change |
| Lead AI Engineer | **Vũ Thế Đệ** | `ENGINEER_MODEL` (e.g. deepseek-r1) | Review feasibility → implement valid train.py |

### Collaboration flow (per experiment)

```
Bùi Đức Toàn (Scientist)
  → Reads history + current train.py
  → Emits: {analysis, hypothesis, proposed_param, proposed_value,
             confidence, risk, reasoning}

Vũ Thế Đệ (Engineer)
  → Receives Scientist proposal + full context
  → Emits: {review, refinement, description, new_code}
  → Produces the final validated train.py
```

---

## Project structure

```
train.py          ← THE ONLY FILE agents edit
app_gradio.py     ← Live UI: agent reasoning, experiment history, log
main.py           ← CLI entry point
config.py         ← Models, agent names, VRAM settings
core/
  pipeline.py     ← Multi-agent experiment loop + event_cb
  history.py      ← results.tsv manager
agent/
  multi_agent.py  ← MultiAgentCoordinator (Toàn → Đệ)
  prompt_builder.py ← Prompts for both agents + legacy single-agent
  llm_client.py   ← OpenRouter HTTP client
  parser.py       ← Robust LLM output parser
executor/
  trainer.py      ← VRAM-safe subprocess runner
  vram_manager.py ← GPU memory cleanup between experiments
  validator.py    ← Syntax + fixed-param + single-change checks
  git_ops.py      ← Branch/commit/rollback
utils/
  metrics.py      ← Parse summary.json / run.log
  plotter.py      ← progress chart
data/
  prepare.py      ← Dataset → YOLO layout converter
```

---

## Configuration

```bash
# Required
export OPENROUTER_API_KEY=...

# Optional: override default models
export SCIENTIST_MODEL=anthropic/claude-3.5-sonnet
export ENGINEER_MODEL=openrouter/deepseek/deepseek-r1

# VRAM tuning (auto-injected into training subprocess)
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
```

---

## Running

### CLI (multi-agent, default)
```bash
python main.py --experiments 10 --gpus 1
```

### CLI (custom models)
```bash
python main.py \
  --scientist-model anthropic/claude-3.5-sonnet \
  --engineer-model  openrouter/deepseek/deepseek-r1 \
  --experiments 10
```

### CLI (legacy single-agent)
```bash
python main.py --model openrouter/meta-llama/llama-3.3-70b-instruct:free
```

### Gradio live demo
```bash
python app_gradio.py        # opens http://localhost:7860
```
The UI shows:
- **Bùi Đức Toàn**'s analysis, hypothesis, confidence, and deep reasoning
- **Vũ Thế Đệ**'s engineering review, refinement, and final description
- Live experiment history table (mAP, VRAM, status, confidence)
- Real-time training log

---

## VRAM management

To prevent cumulative VRAM growth across many experiments:

1. **`train.py` finally block** — explicit `del model`, `torch.cuda.empty_cache()`,
   `torch.cuda.synchronize()`, `reset_peak_memory_stats()` before process exit.

2. **`VRAMManager.training_env()`** — injects
   `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True`
   into the training subprocess to reduce allocator fragmentation.

3. **`VRAMManager.post_run_cooldown()`** — brief sleep after SIGKILL of
   process tree to let the OS fully reclaim the CUDA context.

4. **`Trainer._kill_process_tree()`** — kills the entire process group
   (SIGKILL on Linux) ensuring DDP workers cannot hold the GPU context open.

---

## What Bùi Đức Toàn may propose changing in `train.py`

Only the **hyperparameter block** above `# Do NOT edit below this line`:

| Group | Parameters |
|---|---|
| Schedule | `CLOSE_MOSAIC` |
| Optimiser | `OPTIMIZER`, `LR0`, `LRF`, `MOMENTUM`, `WEIGHT_DECAY` |
| Warmup | `WARMUP_EPOCHS`, `WARMUP_BIAS_LR` |
| Augmentation | `HSV_H/S/V`, `DEGREES`, `TRANSLATE`, `SCALE`, `SHEAR`, `FLIPUD`, `FLIPLR`, `MOSAIC`, `MIXUP`, `COPY_PASTE` |
| Regularisation | `DROPOUT`, `LABEL_SMOOTHING` |

Fixed (Vũ Thế Đệ will reject any change to these):
`MODEL_SIZE`, `PRETRAINED`, `EPOCHS`, `IMGSZ`, `BATCH`, `WORKERS`, `AMP`, `PATIENCE`

---

## 10-Experiment Budget Strategy

### Phase 1 — Baseline (exp 01)
Bùi Đức Toàn lets the default config run as reference. Vũ Thế Đệ makes no changes.

### Phase 2 — Exploration: one change at a time (exp 02–06)
Bùi Đức Toàn proposes from priority list (highest ROI first):
```
1. LR0, LRF         — most impactful for fine-tuning
2. OPTIMIZER        — AdamW vs SGD
3. CLOSE_MOSAIC     — stabilises late training on small data
4. MOSAIC / MIXUP   — reduce if dataset < 200 images
5. WEIGHT_DECAY     — 5e-4 or 1e-3
6. WARMUP_EPOCHS    — 5.0 for more stable start
7. LABEL_SMOOTHING  — 0.1
```
Vũ Thế Đệ validates VRAM budget and rejects/refines if needed.

### Phase 3 — Exploitation: combine & refine (exp 07–10)
Bùi Đức Toàn proposes combining all kept changes.
Vũ Thế Đệ implements the combined config and checks for conflicts.

---

## Output format from `train.py`

```
---
val_mAP5095:      0.452300
val_mAP50:        0.623400
training_epochs:  30
training_seconds: 312.1
peak_vram_mb:     8024.0
checkpoint:       output/train/exp/weights/best.pt
```

## results.tsv format

```
commit  val_mAP5095  val_mAP50  memory_gb  status  description
a1b2c3  0.4123       0.6012     7.8        keep    baseline
b2c3d4  0.4351       0.6210     7.9        keep    REASON: LR too high | CHANGE: LR0=5e-4 | EXPECTED: +0.02
```

**Status values**: `keep` | `discard` | `crash`