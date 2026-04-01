from pathlib import Path
from config import Config

class HistoryManager:
    TSV_HEADER = "commit\tval_mAP5095\tval_mAP50\tmemory_gb\tstatus\tdescription\n"

    def __init__(self):
        self.path = Config.RESULTS_FILE
        self.records = []
        self._ensure_file()

    def _ensure_file(self):
        if not self.path.exists():
            self.path.write_text(self.TSV_HEADER)

    def load(self):
        self.records = []
        if not self.path.exists():
            return
        lines = self.path.read_text().splitlines()[1:]
        for line in lines:
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            self.records.append({
                "commit": parts[0],
                "val_mAP5095": float(parts[1]),
                "val_mAP50": float(parts[2]),
                "memory_gb": float(parts[3]),
                "status": parts[4],
                "description": parts[5],
            })

    def append(self, commit: str, metrics: dict, status: str, description: str):
        mem_gb = round(metrics.get("peak_vram_mb", 0.0) / 1024, 1)
        row = f"{commit}\t{metrics['val_mAP5095']:.6f}\t{metrics['val_mAP50']:.6f}\t{mem_gb}\t{status}\t{description}\n"
        with open(self.path, "a") as f:
            f.write(row)
        self.records.append({
            "val_mAP5095": metrics["val_mAP5095"],
            "status": status,
            "description": description
        })

    def get_completed_count(self) -> int:
        return len(self.records)

    def get_best_map(self) -> float:
        return max((h["val_mAP5095"] for h in self.records), default=0.0)