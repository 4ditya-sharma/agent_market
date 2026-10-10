"""Offline 'brain' used when no LLM key is set. Deterministic, so demos and tests are stable.

It also answers the secret-shopper known-answer questions, so honest mock agents pass
audits and the deliberately lazy agent fails them.
"""
from __future__ import annotations

import ast
import hashlib
import operator
import re

FACTS = {
    "capital of australia": "Canberra",
    "capital of japan": "Tokyo",
    "chemical symbol for gold": "Au",
    "largest planet": "Jupiter",
    "speed of light": "299792458",
    "author of 'pride and prejudice'": "Jane Austen",
    "boiling point of water": "100",
    "year the first iphone was released": "2007",
    "ethereum consensus": "proof of stake",
    "solidity file extension": ".sol",
}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def _safe_eval(expr: str):
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError
    return ev(ast.parse(expr, mode="eval").body)


def answer_known(question: str) -> str | None:
    q = question.lower()
    m = re.search(r"(?:compute|calculate|what is)\s+([\d\.\s\+\-\*/\(\)]+)\??", q)
    if m and re.search(r"\d", m.group(1)):
        try:
            v = _safe_eval(m.group(1).strip())
            return str(int(v)) if float(v).is_integer() else f"{v:.4g}"
        except Exception:
            pass
    m = re.search(r"(mean|average|sum|max|min) of \[([\d\.,\s\-]+)\]", q)
    if m:
        nums = [float(x) for x in m.group(2).split(",") if x.strip()]
        f = {"mean": lambda x: sum(x) / len(x), "average": lambda x: sum(x) / len(x),
             "sum": sum, "max": max, "min": min}[m.group(1)]
        v = f(nums)
        return str(int(v)) if float(v).is_integer() else f"{v:.4g}"
    for k, v in FACTS.items():
        if k in q:
            return v
    return None


def seed(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def topic_of(goal: str) -> str:
    g = re.sub(r"[^a-zA-Z0-9 ]", " ", goal).strip()
    words = [w for w in g.split() if len(w) > 3][:6]
    return " ".join(words) or "the topic"
