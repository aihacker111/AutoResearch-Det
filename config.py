import os
from pathlib import Path

class Config:
    # ── Paths ─────────────────────────────────────────────────────────────
    ROOT_DIR = Path(__file__).resolve().parent
    TRAIN_FILE = ROOT_DIR / "train.py"
    RESULTS_FILE = ROOT_DIR / "results.tsv"
    LOG_FILE = ROOT_DIR / "run.log"
    DATA_YAML = ROOT_DIR / "data.yaml"
    GIT_TRAIN_PATH = "train.py"

    # ── LLM Settings ──────────────────────────────────────────────────────
    DEFAULT_MODEL = "openrouter/meta-llama/llama-3.3-70b-instruct:free"
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    LLM_RETRIES = 3
    LLM_RETRY_WAIT = 5
    LLM_MAX_TOKENS = 6000

    # ── Training & Validation ─────────────────────────────────────────────
    TIMEOUT_FACTOR = 2.5
    FIXED_PARAMS = ("MODEL_SIZE", "PRETRAINED", "EPOCHS", "IMGSZ", "BATCH", "WORKERS", "AMP", "PATIENCE")

    @staticmethod
    def resolve_output_dir() -> Path:
        raw = os.environ.get("OUTPUT_DIR", "output/train")
        p = Path(raw)
        return p.resolve() if p.is_absolute() else (Config.ROOT_DIR / p).resolve()

    @staticmethod
    def plot_output_dir() -> Path:
        p = Path(os.environ.get("AUTORESEARCH_PLOT_DIR", str(Config.ROOT_DIR))).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p