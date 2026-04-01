import re
from config import Config

class Validator:
    @staticmethod
    def sanitize(src: str) -> str:
        replacements = (
            ("\u2018", "'"), ("\u2019", "'"), ("\u201a", "'"), ("\u201b", "'"),
            ("\u201c", '"'), ("\u201d", '"'), ("\u201e", '"'), ("\u201f", '"'),
            ("\u2032", "'"), ("\u2033", '"'), ("\u2014", "-"), ("\u2013", "-"),
            ("\u2012", "-"), ("\u2015", "-"), ("\u2026", "..."), ("\u00a0", " "),
            ("\u200b", ""), ("\u200c", ""), ("\u200d", ""), ("\u2060", ""),
            ("\ufeff", ""), ("\u00b4", "'")
        )
        for bad, good in replacements:
            src = src.replace(bad, good)
        src = src.replace("\r\n", "\n").replace("\r", "\n")
        src = "\n".join(line.rstrip() for line in src.splitlines())
        return src.rstrip("\n") + "\n"

    @staticmethod
    def validate_syntax(src: str):
        s = src.lstrip("\ufeff \t\r\n")
        if not (s.startswith('"""') or s.startswith("'''")):
            raise ValueError(f"train.py must start with docstring. Got: {s[:80]!r}")
        try:
            compile(src, str(Config.TRAIN_FILE), "exec")
        except SyntaxError as e:
            lines = src.splitlines()
            ctx = "\n".join(f"{'>>>' if i+1 == e.lineno else '   '} {i+1:4d} {lines[i]}" for i in range(max(0, e.lineno-4), min(len(lines), e.lineno+2)))
            raise ValueError(f"SyntaxError on line {e.lineno}: {e.msg}\n{ctx}") from e

    @staticmethod
    def validate_fixed_params(new_src: str, orig_src: str):
        param_re = r'^{param}\s*=\s*(.+?)(?:\s*#.*)?$'
        for param in Config.FIXED_PARAMS:
            orig_m = re.search(param_re.format(param=param), orig_src, re.MULTILINE)
            new_m = re.search(param_re.format(param=param), new_src, re.MULTILINE)
            if orig_m and not new_m: raise ValueError(f"LLM removed fixed param {param}")
            if orig_m and new_m and orig_m.group(1).strip() != new_m.group(1).strip():
                raise ValueError(f"LLM changed fixed param {param}")

    @staticmethod
    def validate_single_change(new_src: str, orig_src: str):
        param_re = re.compile(r'^([A-Z0-9_]+)\s*=\s*(.+?)(?:\s*#.*)?$', re.MULTILINE)
        orig_params = dict(param_re.findall(orig_src))
        new_params = dict(param_re.findall(new_src))

        changed = [k for k in set(orig_params) & set(new_params) if orig_params[k].strip() != new_params[k].strip()]
        if len(changed) <= 1: return

        allowed_groups = [
            {"HSV_H", "HSV_S", "HSV_V"},
            {"DEGREES", "TRANSLATE", "SCALE", "SHEAR"},
            {"MOSAIC", "MIXUP", "COPY_PASTE"},
            {"LR0", "LRF"},
            {"WARMUP_EPOCHS", "WARMUP_BIAS_LR"},
            {"DROPOUT", "LABEL_SMOOTHING"},
        ]
        if any(set(changed) <= group for group in allowed_groups): return
        raise ValueError(f"Changed {len(changed)} params: {changed}. Max allowed is 1 hyperparameter or 1 valid group.")