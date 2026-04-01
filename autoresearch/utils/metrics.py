import json
import re
from config import Config

class MetricsParser:
    @staticmethod
    def parse() -> tuple[dict[str, float], bool]:
        metrics = {"val_mAP5095": 0.0, "val_mAP50": 0.0, "peak_vram_mb": 0.0}
        base = Config.resolve_output_dir()
        
        candidates = sorted(base.glob("exp*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in candidates:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                metrics["val_mAP5095"] = float(data.get("val_mAP5095", 0.0))
                metrics["val_mAP50"] = float(data.get("val_mAP50", 0.0))
                metrics["peak_vram_mb"] = float(data.get("peak_vram_mb", 0.0))
                return metrics, True
            except: pass

        log_metrics, log_ok = MetricsParser._parse_from_log()
        if log_ok: return log_metrics, True
        return metrics, False

    @staticmethod
    def _parse_from_log() -> tuple[dict[str, float], bool]:
        metrics = {"val_mAP5095": 0.0, "val_mAP50": 0.0, "peak_vram_mb": 0.0}
        try: text = Config.LOG_FILE.read_text(errors="ignore")
        except FileNotFoundError: return metrics, False
        
        for line in text.splitlines():
            for key in metrics:
                m = re.match(rf"^{re.escape(key)}:\s+([\d.]+)", line.strip())
                if m: metrics[key] = float(m.group(1))
        
        ok = re.search(r"^val_mAP5095:\s+[\d.]+", text, re.MULTILINE) is not None
        return metrics, ok