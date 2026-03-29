# autoresearch-det

> Autonomous hyperparameter search for YOLO11 object detection fine-tuning.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
An LLM agent proposes one hyperparameter change per iteration, trains, measures
`val/mAP50-95`, keeps improvements, discards regressions — overnight, unattended.

---

## How it works

```
run_autoresearch.py          ← orchestrator (you don't touch)
│
├── calls LLM  →  "change LR0 from 1e-3 to 5e-4"
├── edits train.py
├── git commit
├── python train.py > run.log
├── grep val_mAP5095
├── keep  → advance branch
└── discard → git reset
```

**Only two files matter to the researcher:**

| File | Role |
|---|---|
| `train.py` | Agent edits this — all hyperparameters |
| `program.md` | You edit this — research goals & constraints |

---

## Quick start

```bash
# 1. Install
pip install ultralytics matplotlib pyyaml

# 2. Prepare dataset (YOLO or COCO format)
export DATASET_DIR=/path/to/your/dataset
python prepare.py --dataset-dir $DATASET_DIR

# 3. Set API key (free tier: openrouter.ai)
export OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxx

# 4. Run 10 experiments autonomously
python run_autoresearch.py --experiments 10
```

---

## Dataset formats

`prepare.py` auto-detects and generates `data.yaml` for:

| Format | Structure |
|---|---|
| YOLO flat | `images/` + `labels/` + `data.yaml` |
| Roboflow | `train/valid/test` + `_annotations.coco.json` |
| COCO JSON | `annotations/*.json` |
| Custom | Any layout with a `classes.txt` |

---

## Multi-GPU

```bash
python run_autoresearch.py --experiments 10 --cuda-devices 0,1
```

---

## Monitor progress

```bash
# Live log
tail -f run.log

# Results table
cat results.tsv

# Charts (generated automatically after each experiment)
# progress.png          — bar chart per experiment
# progress_detail.png   — step chart of improvements only
python plot_progress.py
```

---

## Supported LLM models (OpenRouter)

```bash
# Free tier
--model openrouter/meta-llama/llama-3.3-70b-instruct:free

# Paid (better code quality)
--model openrouter/anthropic/claude-sonnet-4-5
--model openrouter/openai/gpt-4.1-mini
```

---

## Project structure

```
autoresearch-det/
├── train.py              ← agent edits this
├── prepare.py            ← dataset setup + eval harness  [fixed]
├── run_autoresearch.py   ← experiment loop               [fixed]
├── plot_progress.py      ← progress charts               [fixed]
├── program.md            ← agent instructions            [you edit]
├── pyproject.toml        ← dependencies                  [fixed]
├── data.yaml             ← auto-generated
└── results.tsv           ← experiment log (not committed)
```

---

## License

MIT