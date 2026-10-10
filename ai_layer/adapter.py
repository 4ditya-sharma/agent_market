"""Adapter wrapper: makes ANY agent look the same to the marketplace.

A marketplace agent only needs `card` + `run(subtask, context) -> WorkerResult`.
Three ways to plug one in:
  1. Subclass BaseWorker (our demo workers).
  2. FunctionAdapter: wrap a plain Python function  fn(instructions, context) -> str | dict.
  3. HTTPAdapter: wrap a remote agent that accepts POST {"instructions", "context"}
     and returns {"output": "...", "artifacts": [...], "data": {...}}.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from .schemas import AgentCard, Subtask, WorkerResult


class BaseWorker:
    card: AgentCard

    def handle(self, subtask: Subtask, context: dict[str, WorkerResult]) -> dict[str, Any]:
        """Return dict with keys: output (str), optional artifacts, data, source."""
        raise NotImplementedError

    def run(self, subtask: Subtask, context: dict[str, WorkerResult] | None = None) -> WorkerResult:
        t0 = time.time()
        try:
            res = self.handle(subtask, context or {})
        except Exception as e:  # never let one bad agent crash the run
            res = {"output": f"[agent error] {type(e).__name__}: {e}", "source": "error"}
        return WorkerResult(
            subtask_id=subtask.id,
            agent_id=self.card.agent_id,
            output=str(res.get("output", "")),
            artifacts=res.get("artifacts", []),
            data=res.get("data", {}),
            source=res.get("source", "mock"),
            latency_ms=int((time.time() - t0) * 1000),
        )


class FunctionAdapter(BaseWorker):
    def __init__(self, card: AgentCard, fn: Callable[[str, dict], Any]):
        self.card, self.fn = card, fn

    def handle(self, subtask, context):
        ctx = {k: v.output for k, v in context.items()}
        out = self.fn(subtask.instructions, ctx)
        return out if isinstance(out, dict) else {"output": str(out), "source": "function"}


class HTTPAdapter(BaseWorker):
    def __init__(self, card: AgentCard, timeout: float = 60):
        if not card.endpoint:
            raise ValueError("HTTPAdapter needs card.endpoint")
        self.card, self.timeout = card, timeout

    def handle(self, subtask, context):
        import httpx

        payload = {
            "subtask": subtask.model_dump(),
            "instructions": subtask.instructions,
            "context": {k: v.output for k, v in context.items()},
        }
        r = httpx.post(self.card.endpoint, json=payload, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        body.setdefault("source", f"http:{self.card.endpoint}")
        return body
