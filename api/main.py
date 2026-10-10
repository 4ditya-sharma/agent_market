"""FastAPI service the backend (Karan) calls. Run:  uvicorn api.main:app --port 8001 --reload
Docs at http://localhost:8001/docs

Every endpoint is JSON in / JSON out and stateless except the in-memory registry + trust scores,
which the backend should mirror into its own DB (see README "Backend contract").
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai_layer import build_marketplace
from ai_layer.adapter import HTTPAdapter
from ai_layer.audit import audit_agent
from ai_layer.firewall import get_firewall
from ai_layer.schemas import AgentCard, AuditReport, RunResult

app = FastAPI(title="AgentMarket AI layer", version="0.1.0")
registry, workers, master = build_marketplace(include_bad_actors=os.getenv("DEMO_BAD_ACTORS", "1") == "1")


class RunRequest(BaseModel):
    goal: str = Field(..., examples=["Market analysis deck on AI agent marketplaces in India"])
    budget: float = Field(20.0, gt=0)
    buyer_id: Optional[str] = None


class ClassifyRequest(BaseModel):
    text: str
    agent_id: Optional[str] = None


class SearchResponse(BaseModel):
    results: list[dict]


@app.get("/health")
def health():
    fw = get_firewall()
    return {"ok": True, "firewall": fw.backend, "firewall_threshold": fw.threshold,
            "registry_backend": registry.backend, "agents": len(registry.agents)}


@app.post("/run", response_model=RunResult)
def run(req: RunRequest):
    """Full pipeline: plan -> hire -> execute -> firewall -> judge -> payouts."""
    return master.run(req.goal, req.budget)


@app.post("/classify")
def classify(req: ClassifyRequest):
    """Firewall only. severity: benign | malicious | malignant."""
    return get_firewall().score_text(req.text)


@app.get("/agents/search", response_model=SearchResponse)
def search(q: str, top_k: int = 5, capability: Optional[str] = None, max_price: Optional[float] = None):
    return {"results": registry.search(q, top_k=top_k, capability=capability, max_price=max_price)}


@app.get("/agents")
def list_agents():
    return [{"card": c.model_dump(), "trust": registry.trust[c.agent_id]} for c in registry.agents.values()]


@app.post("/agents/register")
def register(card: AgentCard):
    """Register an external HTTP agent. card.endpoint must accept the adapter POST contract."""
    if not card.endpoint:
        raise HTTPException(400, "external agents need an endpoint")
    w = HTTPAdapter(card)
    workers[card.agent_id] = w
    registry.register(card, trust=0.5)
    return {"registered": card.agent_id, "trust": 0.5}


@app.post("/audit/{agent_id}", response_model=AuditReport)
def audit(agent_id: str, n_tasks: int = 6):
    """Secret-shopper audit; also updates the agent's trust score in the registry."""
    if agent_id not in workers:
        raise HTTPException(404, "unknown agent")
    rep = audit_agent(workers[agent_id], n_tasks=n_tasks)
    registry.trust[agent_id] = rep.trust_score
    return rep


@app.get("/firewall/quarantine")
def quarantine():
    return get_firewall().quarantine
