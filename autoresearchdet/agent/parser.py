import re
import json

class LLMParser:
    UNICODE_REPLACEMENTS = (
        ("\u2018", "'"), ("\u2019", "'"), ("\u201a", "'"), ("\u201b", "'"),
        ("\u201c", '"'), ("\u201d", '"'), ("\u201e", '"'), ("\u201f", '"'),
        ("\u2032", "'"), ("\u2033", '"'), ("\u2014", "-"), ("\u2013", "-"),
        ("\u2012", "-"), ("\u2015", "-"), ("\u2026", "..."),("\u00a0", " "),
        ("\u200b", ""), ("\u200c", ""), ("\u200d", ""), ("\u2060", ""),
        ("\ufeff", ""), ("\u00b4", "'")
    )

    @classmethod
    def parse(cls, raw_text: str) -> dict:
        text = cls._normalize_text(raw_text)
        text = cls._strip_outer_markdown(text)

        # Execute 10-stage parser precisely
        for fn in (cls._try_json_raw, cls._try_json_loads_whole):
            out = fn(text)
            if out: return out

        fixed_nl = cls._fix_literal_newlines(text)
        if fixed_nl != text:
            for fn in (cls._try_json_raw, cls._try_json_loads_whole):
                out = fn(fixed_nl)
                if out: return out

        for fn in (cls._try_fenced_json, cls._try_triple_quote, cls._try_regex_fields, cls._try_greedy):
            out = fn(text)
            if out: return out

        raw_normalized = cls._normalize_text(raw_text)
        for fn in (cls._try_markdown_blocks, cls._try_plain_python):
            out = fn(raw_normalized) or fn(text)
            if out: return out

        raise ValueError(f"Cannot parse LLM response -- all parser stages failed.\nFirst 600 chars:\n{raw_normalized[:600]}")

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        t = text.strip()
        for bad, good in cls.UNICODE_REPLACEMENTS:
            t = t.replace(bad, good)
        return t.encode("utf-8", errors="replace").decode("utf-8")

    @staticmethod
    def _strip_outer_markdown(text: str) -> str:
        m = re.match(r"^```\S*\s*\r?\n(.*)\r?\n```\s*$", text.strip(), re.DOTALL)
        return m.group(1).strip() if m else text.strip()

    @staticmethod
    def _looks_like_train_py(s: str) -> bool:
        if not s or len(s.strip()) < 200: return False
        has_py = "\ndef " in s or s.startswith("def ") or '"""' in s or "import " in s
        if not has_py: return False
        return any(k in s for k in ["MODEL_SIZE", "EPOCHS", "ultralytics", "YOLO", "model.train("])

    @staticmethod
    def _unescape(s: str) -> str:
        try: return json.loads('"' + s + '"')
        except: return s.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\")

    @staticmethod
    def _try_json_raw(text: str) -> dict | None:
        i = text.find("{")
        if i < 0: return None
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, i)
            if isinstance(obj, dict) and "new_code" in obj and obj["new_code"]:
                return {"description": str(obj.get("description", "LLM proposal")).strip(), "new_code": str(obj["new_code"])}
        except: pass
        return None

    @staticmethod
    def _try_json_loads_whole(text: str) -> dict | None:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "new_code" in obj and obj["new_code"]:
                return {"description": str(obj.get("description", "LLM proposal")).strip(), "new_code": str(obj["new_code"])}
        except: pass
        return None

    @staticmethod
    def _fix_literal_newlines(text: str) -> str:
        start_m = re.search(r'"new_code"\s*:\s*"', text)
        if not start_m: return text
        prefix, rest = text[:start_m.end()], text[start_m.end():]
        fixed, i, closed = [], 0, False
        while i < len(rest):
            c = rest[i]
            if c == "\\" and i + 1 < len(rest): fixed.append(rest[i:i+2]); i += 2
            elif c == '"': fixed.append('"'); i += 1; closed = True; break
            elif c == "\r" and i + 1 < len(rest) and rest[i+1] == "\n": fixed.append("\\n"); i += 2
            elif c in ("\n", "\r"): fixed.append("\\n"); i += 1
            else: fixed.append(c); i += 1
        return prefix + "".join(fixed) + rest[i:] if closed else text

    @staticmethod
    def _try_fenced_json(text: str) -> dict | None:
        m = re.search(r"```(?:json)?\s*\r?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if not m: return None
        inner = m.group(1).strip()
        return LLMParser._try_json_raw(inner) or LLMParser._try_json_loads_whole(inner)

    @staticmethod
    def _try_triple_quote(text: str) -> dict | None:
        start_m = re.search(r'"new_code"\s*:\s*"""', text, re.DOTALL)
        if not start_m: return None
        after = text[start_m.end():]
        triples = [m.start() for m in re.finditer(r'"""', after)]
        if not triples: return None
        new_code = after[:triples[-1]].strip()
        if not new_code: return None
        desc_m = re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        desc = LLMParser._unescape(desc_m.group(1)) if desc_m else "LLM proposal (triple-quoted)"
        return {"description": desc, "new_code": new_code}

    @staticmethod
    def _try_regex_fields(text: str) -> dict | None:
        desc_m = re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        code_m = re.search(r'"new_code"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
        if desc_m and code_m:
            body = LLMParser._unescape(code_m.group(1))
            if body.strip(): return {"description": LLMParser._unescape(desc_m.group(1)), "new_code": body}
        return None

    @staticmethod
    def _try_greedy(text: str) -> dict | None:
        start_m = re.search(r'"new_code"\s*:\s*"', text)
        if not start_m: return None
        raw = text[start_m.end():]
        raw_code = raw.rstrip('"}\n\r ')
        for pat in [r'"\s*\n\s*\}\s*$', r'"\s*\}\s*$', r'"\s*$']:
            end_m = re.search(pat, raw)
            if end_m: raw_code = raw[:end_m.start()]; break
        code = LLMParser._unescape(raw_code).replace("\r\n", "\n").replace("\r", "\n")
        if LLMParser._looks_like_train_py(code):
            return {"description": "LLM proposal (greedy)", "new_code": code}
        return None

    @staticmethod
    def _try_markdown_blocks(text: str) -> dict | None:
        blocks = re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL) or re.findall(r"```[^\n]*(.*?)```", text, re.DOTALL)
        best = next((b.strip() for b in sorted(blocks, key=len, reverse=True) if LLMParser._looks_like_train_py(b)), None)
        if best:
            m = re.search(r"(?:description|change|summary)\s*[:#]\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
            return {"description": m.group(1).strip()[:200] if m else "LLM proposal (code block)", "new_code": best}
        return None

    @staticmethod
    def _try_plain_python(text: str) -> dict | None:
        t = text.strip()
        if not t.startswith("{") and not t.startswith("[") and LLMParser._looks_like_train_py(t):
            for line in t.splitlines():
                cl = line.strip().lstrip("#").strip().strip('"').strip("'").strip()
                if len(cl) > 5: return {"description": cl[:120], "new_code": t}
            return {"description": "Raw train.py", "new_code": t}
        return None