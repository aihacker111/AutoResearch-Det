import argparse
import os
import sys
from config          import Config
from core.pipeline    import AutoResearchPipeline


def main():
    parser = argparse.ArgumentParser(
        description="AutoResearch-DET: multi-agent YOLO11 hyperparameter search"
    )
    parser.add_argument("--experiments",     type=int, default=10)
    # ── Multi-agent models (default path) ─────────────────────────────────
    parser.add_argument("--scientist-model", default=None,
                        help="LLM for Bùi Đức Toàn (Scientist). "
                             f"Default: {Config.SCIENTIST_MODEL}")
    parser.add_argument("--engineer-model",  default=None,
                        help="LLM for Vũ Thế Đệ (Engineer). "
                             f"Default: {Config.ENGINEER_MODEL}")
    # ── Legacy single-agent override ──────────────────────────────────────
    parser.add_argument("--model",           default=None,
                        help="Use single-agent mode with this model (overrides multi-agent)")
    # ── Hardware ──────────────────────────────────────────────────────────
    parser.add_argument("--gpus",            type=int, default=0)
    parser.add_argument("--cuda-devices",    default=None)
    # ── Misc ──────────────────────────────────────────────────────────────
    parser.add_argument("--resume",          action="store_true")
    parser.add_argument("--dry-run",         action="store_true")
    parser.add_argument("--quiet",           action="store_true")
    args = parser.parse_args()

    if not Config.OPENROUTER_API_KEY:
        sys.exit("ERROR: set OPENROUTER_API_KEY environment variable")

    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Merge CLI model overrides into args so pipeline can read them
    if args.scientist_model:
        args.scientist_model = args.scientist_model
    else:
        args.scientist_model = Config.SCIENTIST_MODEL

    if args.engineer_model:
        args.engineer_model = args.engineer_model
    else:
        args.engineer_model = Config.ENGINEER_MODEL

    pipeline = AutoResearchPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()