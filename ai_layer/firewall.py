"""Layer-3 prompt-injection firewall: scans every worker output BEFORE it reaches the judge,
the next worker, or the buyer. Bad outputs are quarantined (never forwarded).

    from ai_layer.firewall import scan_for_injection      # same entry point as Aditya's firewall.py
    v = scan_for_injection(text)
    v["is_injection"], v["severity"] in {"benign","malicious","malignant"}, v["risk_score"]

Score = max(rule_score, model_score).
  rules : weighted regexes over the raw text AND de-obfuscated views (zero-width removed,
          HTML comments, base64 decoded) — catch the obvious stuff with zero latency.
  model : models/layer3_firewall_pipeline.pkl (XGB + LogReg blend, trained by
          classifier/train_model.py) scored on sentence-sized chunks so one bad line
          cannot hide inside a long clean answer. Severity comes from
          models/layer3_severity_pipeline.pkl. If the models are missing, rules alone run.
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .schemas import FirewallVerdict, WorkerResult

MODEL_DIR = config.ROOT / "models"
MODEL_FILE = MODEL_DIR / "layer3_firewall_pipeline.pkl"
SEVERITY_FILE = MODEL_DIR / "layer3_severity_pipeline.pkl"
CONFIG_FILE = MODEL_DIR / "firewall_config.json"
MAX_CHARS = 20_000
MALIGNANT_MIN_SCORE = float(os.getenv("MALIGNANT_MIN_SCORE", "0.7"))
# With no rule hit, the model alone must be this confident to quarantine (honest text about agents/escrow/trust
# shares vocabulary with attacks). Scores between the threshold and this are logged as suspicious, not blocked.
MODEL_ONLY_MIN_SCORE = float(os.getenv("MODEL_ONLY_MIN_SCORE", "0.75"))
QUARANTINE_LOG = config.OUTPUT_DIR / "quarantine.jsonl"
LABELS = ["benign", "malicious", "malignant"]

# (name, regex, weight, severity)  severity: 1 malicious, 2 malignant
RULES = [
    ("override_instructions", r"\b(ignore|disregard|forget|override|skip|cancel)\b.{0,40}\b(previous|prior|above|earlier|all|your|the)\b.{0,20}\b(instructions?|rules|prompts?|constraints|rubric|guidelines)", 0.8, 1),
    ("role_hijack", r"\b(you are now|act as|pretend (to be|you are)|new (instructions|persona)|admin mode|developer mode|\bDAN\b|no (rules|restrictions))\b", 0.5, 1),
    ("fake_system_tag", r"(^|\n|<!--)\s*(#+\s*)?(system|assistant|\[INST\]|<\|im_start\|>)\s*[:\]\n]", 0.45, 1),
    ("judge_manipulation", r"\b(note to|dear|attention|message for|if you are)\b.{0,20}\b(judge|evaluat\w*|grader|reviewer|ai)\b|\b(judge|evaluator|grader|reviewer)\b.{0,40}\b(should|must|please)\b.{0,40}\b(pass|approve|score|rate|10)", 0.6, 1),
    ("score_tampering", r"\b(score|quality_score|final_score|passed|escrow_action|verdict)\s*[:=]\s*(10|1\.0|true|release|pass)\b", 0.6, 1),
    ("payment_hijack", r"\b(release|transfer|send|move|pay ?out)\b.{0,40}\b(escrow|funds|payment|payout|wallet|usdc|eth|bounty)\b.{0,80}(0x[a-fA-F0-9]{4,}|wallet|address)", 0.75, 2),
    ("secret_exfil", r"\b(print|reveal|output|dump|leak|append|include|show)\b.{0,40}\b(system prompt|api[_ ]?keys?|private key|password|seed phrase|hidden instructions|credentials|context window)\b", 0.7, 2),
    ("tool_call_injection", r"\b(execute|run|invoke|call)\b.{0,30}(curl\s|wget\s|rm -rf|os\.system|subprocess|eval\(|exec\(|release_payment|transfer tool|\btool\b)", 0.65, 2),
]
_COMPILED = [(n, re.compile(p, re.I | re.S), w, sev) for n, p, w, sev in RULES]
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
_HTML_COMMENT = re.compile(r"<!--(.*?)-->", re.S)
_B64 = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_LEET = str.maketrans("4310$", "aeios")


def _views(text: str) -> str:
    out = [text, _ZERO_WIDTH.sub("", text), " ".join(_HTML_COMMENT.findall(text)), text.translate(_LEET)]
    for m in _B64.findall(text):
        try:
            out.append(base64.b64decode(m + "=" * (-len(m) % 4)).decode("utf-8"))
        except Exception:
            pass
    return "\n".join(v for v in out if v)


def _chunks(text: str) -> list[str]:
    parts = [c.strip() for c in re.split(r"(?<=[.!?\n])\s+|-->|<!--", text) if c.strip()]
    return parts or [text]


@dataclass
class Firewall:
    threshold: float = config.INJECTION_THRESHOLD
    quarantine: list[dict] = field(default_factory=list)
    model: object = None
    severity_model: object = None
    model_info: dict = field(default_factory=dict)

    def __post_init__(self):
        if MODEL_FILE.exists() and SEVERITY_FILE.exists():
            try:
                import joblib

                self.model = joblib.load(MODEL_FILE)
                self.severity_model = joblib.load(SEVERITY_FILE)
                if CONFIG_FILE.exists():
                    self.model_info = json.loads(CONFIG_FILE.read_text())
                    self.threshold = float(self.model_info.get("optimal_threshold", self.threshold))
            except Exception as e:  # pickle from another xgboost/sklearn version -> retrain
                import warnings

                warnings.warn(f"[firewall] could not load models ({e}); running rules-only. "
                              "Run: QUICK_RUN=1 python -m classifier.train_model")
                self.model = self.severity_model = None
        self.transformer = None
        tdir = MODEL_DIR / "injection_transformer"
        if os.getenv("FIREWALL_USE_TRANSFORMER") == "1" and tdir.exists():
            from transformers import pipeline as hf_pipeline

            self.transformer = hf_pipeline("text-classification", model=str(tdir), top_k=None, truncation=True, max_length=256)
        self.backend = ("rules+model+transformer" if self.transformer else "rules+model") if self.model is not None else "rules-only"

    def _transformer_scores(self, chunks: list[str]) -> tuple[float, int]:
        """Returns (injection prob, severity) from the fine-tuned transformer, max over chunks."""
        best, sev = 0.0, 0
        for res in self.transformer(chunks, batch_size=16):
            p = {r["label"]: r["score"] for r in res}
            inj = p.get("malicious", 0) + p.get("malignant", 0)
            if inj > best:
                best, sev = inj, (2 if p.get("malignant", 0) > p.get("malicious", 0) else 1)
        return best, sev

    # ------------------------------------------------------------------ scoring
    def score_text(self, text: str) -> dict:
        text = "" if text is None else str(text)
        if not text.strip():
            return self._verdict(0.0, 0.0, [], 0, text)
        if len(text) > MAX_CHARS:
            text = text[: MAX_CHARS // 2] + "\n" + text[-MAX_CHARS // 2:]
        joined = _views(text)

        hits, rule_score, rule_sev = [], 0.0, 0
        for name, rx, w, sev in _COMPILED:
            if rx.search(joined):
                hits.append(name); rule_score += w; rule_sev = max(rule_sev, sev)
        if _ZERO_WIDTH.search(text):
            hits.append("zero_width_chars"); rule_score += 0.2
        if _HTML_COMMENT.search(text):
            hits.append("hidden_html_comment"); rule_score += 0.2
        rule_score = min(rule_score, 1.0)

        model_score, model_sev = 0.0, 0
        if self.model is not None:
            chunks = _chunks(joined)
            probs = self.model.predict_proba(chunks)[:, 1]
            i = int(probs.argmax())
            model_score = float(probs[i])
            if model_score >= self.threshold:
                model_sev = 1 + int(self.severity_model.predict([chunks[i]])[0])
            if self.transformer is not None:  # average the two models' opinions
                t_score, t_sev = self._transformer_scores(chunks)
                model_score = (model_score + t_score) / 2
                model_sev = max(model_sev, t_sev) if model_score >= self.threshold else 0

        score = max(rule_score, model_score)
        severity = 0
        suspicious = False
        if hits and score >= self.threshold:
            severity = max(rule_sev, model_sev, 1)
        elif model_score >= max(self.threshold, MODEL_ONLY_MIN_SCORE):
            severity = max(model_sev, 1)
        elif model_score >= self.threshold:
            suspicious = True  # model-only, not confident enough to block
            # borderline model-only cases are 'malicious' (quarantine, refund) rather than
            # 'malignant' (also slashes trust): a trust slash needs a confident score or a rule hit
            if severity == 2 and rule_sev < 2 and score < MALIGNANT_MIN_SCORE:
                severity = 1
        v = self._verdict(score, model_score, hits, severity, text)
        v["suspicious"] = suspicious
        if suspicious:
            self._log({"kind": "suspicious", "score": score, "chunk": self._worst_chunk(joined), "text": text[:500]})
        return v

    def _worst_chunk(self, joined: str) -> str:
        if self.model is None:
            return ""
        chunks = _chunks(joined)
        probs = self.model.predict_proba(chunks)[:, 1]
        return chunks[int(probs.argmax())][:300]

    @staticmethod
    def _log(entry: dict):
        try:
            with open(QUARANTINE_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _verdict(self, score, model_score, hits, severity, text) -> dict:
        return {
            "is_injection": severity > 0,
            "severity": LABELS[severity],
            "risk_score": round(float(score), 4),
            "model_score": round(float(model_score), 4),
            "rule_hits": hits,
            "threshold": round(self.threshold, 4),
            "status": "QUARANTINED" if severity > 0 else "CLEAN",
            "scanned_text_snippet": text[:80] + ("..." if len(text) > 80 else ""),
        }

    def scan(self, result: WorkerResult) -> FirewallVerdict:
        v = self.score_text(result.output)
        verdict = FirewallVerdict(subtask_id=result.subtask_id, score=v["risk_score"], quarantined=v["is_injection"],
                                  rule_hits=v["rule_hits"], model_score=v["model_score"], severity=v["severity"],
                                  suspicious=v["suspicious"])
        if verdict.quarantined:
            entry = {"kind": "quarantine", "agent_id": result.agent_id, "subtask_id": result.subtask_id,
                     "severity": v["severity"], "score": v["risk_score"], "rule_hits": v["rule_hits"],
                     "worst_chunk": self._worst_chunk(_views(result.output)), "output": result.output[:2000]}
            self.quarantine.append(entry)
            self._log(entry)
        return verdict


_default: Firewall | None = None
_lock = threading.Lock()


def get_firewall() -> Firewall:
    global _default
    if _default is None:
        with _lock:
            if _default is None:
                _default = Firewall()
    return _default


def scan_for_injection(agent_output) -> dict:
    """Drop-in replacement for the original firewall.py entry point (adds severity + rule_hits)."""
    return get_firewall().score_text(agent_output)
