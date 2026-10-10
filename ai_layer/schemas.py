"""Shared data shapes. These are also the JSON contract with the backend."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Capability = Literal["research", "analysis", "charts", "deck", "summary"]


class AgentCard(BaseModel):
    """How a worker agent advertises itself in the marketplace."""
    agent_id: str
    name: str
    description: str
    capabilities: list[str]
    price_per_task: float = Field(description="Base price in marketplace credits / USDC")
    owner_wallet: str = "0x0000000000000000000000000000000000000000"
    claimed_quality: float = Field(0.9, ge=0, le=1, description="Quality the agent claims (0-1)")
    endpoint: Optional[str] = None  # set for external HTTP agents


class Subtask(BaseModel):
    id: str
    title: str
    instructions: str
    capability: Capability
    depends_on: list[str] = []
    budget: float = 0.0
    acceptance_criteria: list[str] = []
    assigned_agent: Optional[str] = None


class Plan(BaseModel):
    goal: str
    total_budget: float
    subtasks: list[Subtask]
    source: str = "mock"


class WorkerResult(BaseModel):
    subtask_id: str
    agent_id: str
    output: str
    artifacts: list[str] = []  # file paths (charts, decks)
    data: dict[str, Any] = {}
    source: str = "mock"
    latency_ms: int = 0


class FirewallVerdict(BaseModel):
    subtask_id: str
    score: float  # 0 = clean, 1 = certainly an injection
    quarantined: bool
    rule_hits: list[str] = []
    model_score: float = 0.0
    severity: Literal["benign", "malicious", "malignant"] = "benign"
    suspicious: bool = False  # model score above threshold but below the model-only bar; passed on, logged


class JudgeVote(BaseModel):
    judge: str
    score: float  # 0-10
    passed: bool
    reason: str
    source: str = "mock"


class JudgeVerdict(BaseModel):
    subtask_id: str
    tier1_passed: bool
    tier1_failures: list[str] = []
    votes: list[JudgeVote] = []
    final_score: float = 0.0
    passed: bool = False
    # What the backend should do with the escrow for this subtask
    escrow_action: Literal["release", "refund", "hold"] = "hold"


class RunResult(BaseModel):
    run_id: str
    plan: Plan
    results: list[WorkerResult]
    firewall: list[FirewallVerdict]
    verdicts: list[JudgeVerdict]
    final_output: str
    payouts: dict[str, float]  # agent_id -> amount to release
    attempts: list[dict[str, Any]] = []  # every hire incl. failed/quarantined ones (for audit trail + slashing)


class AuditReport(BaseModel):
    agent_id: str
    tasks_run: int
    audited_accuracy: float
    claimed_quality: float
    gap: float  # claimed - audited (positive = agent overclaims)
    trust_score: float
    flagged: bool
    details: list[dict[str, Any]] = []
