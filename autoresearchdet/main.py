import argparse
import os
import sys
from config import Config
from core.pipeline import AutoResearchPipeline

def main():
    parser = argparse.ArgumentParser(description="AutoResearch-DET: autonomous YOLO11 hyperparameter search")
    parser.add_argument("--experiments", type=int, default=10)
    parser.add_argument("--model", default=Config.DEFAULT_MODEL)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--cuda-devices", default=None)
    parser.add_argument("--resume", action="store_true", help="Continue from existing results.tsv")
    parser.add_argument("--dry-run", action="store_true", help="LLM proposals only, no training")
    parser.add_argument("--quiet", action="store_true", help="Training logs only in run.log")
    args = parser.parse_args()

    if not Config.OPENROUTER_API_KEY:
        sys.exit("ERROR: set OPENROUTER_API_KEY environment variable")

    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    pipeline = AutoResearchPipeline(args)
    pipeline.run()

if __name__ == "__main__":
    main()