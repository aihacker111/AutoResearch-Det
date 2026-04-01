import signal
import sys
from config import Config
from core.history import HistoryManager
from agent.llm_client import LLMClient
from agent.parser import LLMParser
from agent.prompt_builder import PromptBuilder
from executor.git_ops import GitManager
from executor.trainer import Trainer
from executor.validator import Validator
from utils.metrics import MetricsParser
from utils.plotter import Plotter

class AutoResearchPipeline:
    def __init__(self, args):
        self.args = args
        self.history = HistoryManager()
        self.llm = LLMClient(model=args.model)
        self.trainer = Trainer(num_gpus=args.gpus, cuda_devices=args.cuda_devices, quiet=args.quiet)
        self.baseline_time = None
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, sig, frame):
        print("\n[!] Interrupt received -- finishing current experiment then stopping.")
        self.shutdown_requested = True

    def run(self):
        branch = GitManager.setup_branch()
        if not Config.TRAIN_FILE.is_file():
            sys.exit(f"ERROR: {Config.TRAIN_FILE} not found.")

        if self.args.resume:
            self.history.load()

        completed = self.history.get_completed_count()
        best_map = self.history.get_best_map()

        print(f"\n{'='*62}")
        print(f"  AutoResearch-DET  |  YOLO11 fine-tuning")
        print(f"  Branch      : {branch}")
        print(f"  LLM model   : {self.args.model}")
        print(f"  GPUs        : {self.trainer.num_gpus}")
        print(f"  Experiments : {self.args.experiments - completed} remaining / {self.args.experiments} total")
        print(f"  Fixed params: {', '.join(Config.FIXED_PARAMS)}")
        if self.args.dry_run:
            print(f"  Mode        : DRY-RUN (no training)")
        if self.args.quiet:
            print(f"  Training log: quiet (see {Config.LOG_FILE})")
        print(f"{'='*62}\n")

        for exp_num in range(completed + 1, self.args.experiments + 1):
            if self.shutdown_requested:
                print("\n[!] Graceful shutdown -- results saved.")
                break

            print(f"\n{'='*62}")
            print(f"  Experiment {exp_num:02d} / {self.args.experiments:02d}")
            print(f"{'='*62}")

            current_code = Config.TRAIN_FILE.read_text(errors="ignore")

            if exp_num == 1:
                description = "baseline -- default hyperparameters"
                print(f"  Idea  : {description}")
            else:
                print("  LLM   : proposing next experiment...")
                try:
                    sys_prompt, user_prompt = PromptBuilder.build(self.history.records, current_code)
                    raw_response = self.llm.generate(sys_prompt, user_prompt)
                    proposal = LLMParser.parse(raw_response)
                    description = proposal["description"]
                    
                    if not self.args.dry_run:
                        clean_code = Validator.sanitize(proposal["new_code"])
                        Validator.validate_syntax(clean_code)
                        Validator.validate_fixed_params(clean_code, current_code)
                        Validator.validate_single_change(clean_code, current_code)
                        Config.TRAIN_FILE.write_text(clean_code, encoding="utf-8")
                    print(f"  Idea  : {description}")
                except Exception as exc:
                    print(f"  LLM error: {exc} -- skipping")
                    continue

            if self.args.dry_run:
                print("  [dry-run] skipping training")
                self.history.append("dryrun", {"val_mAP5095":0.0, "val_mAP50":0.0, "peak_vram_mb":0.0}, "dry-run", description)
                continue

            commit = GitManager.commit(f"experiment: {description}")
            timeout = self.baseline_time * Config.TIMEOUT_FACTOR if self.baseline_time else None
            print(f"  Commit: {commit}  |  timeout: {f'{timeout:.0f}s' if timeout else 'none'}")

            print("  Training...\n")
            sys.stdout.flush()
            success, elapsed = self.trainer.run(timeout)
            
            if exp_num == 1:
                self.baseline_time = elapsed

            metrics, metrics_ok = MetricsParser.parse()
            val_map = metrics["val_mAP5095"]

            print()
            if not metrics_ok:
                print(f"  X  CRASH  ({elapsed:.0f}s)  -- see {Config.LOG_FILE}")
                self.history.append(commit, metrics, "crash", description)
                GitManager.rollback()
                continue

            if not success:
                print("  [!] Training exited non-zero but metrics recovered. Continuing.")

            prev = self.history.records[-1]["val_mAP5095"] if self.history.records else 0.0
            print(f"  mAP50-95={val_map:.4f}  mAP50={metrics['val_mAP50']:.4f}  VRAM={metrics['peak_vram_mb']/1024:.1f}GB  time={elapsed:.0f}s")

            if val_map > best_map:
                best_map = val_map
                status = "keep"
                print(f"  v  KEEP   (delta={val_map - prev:+.4f})")
            else:
                status = "discard"
                print(f"  X  DISCARD (delta={val_map - prev:+.4f})")
                if exp_num > 1:
                    GitManager.rollback()

            self.history.append(commit, metrics, status, description)
            Plotter.update_plot()