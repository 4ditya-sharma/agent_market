"""Master agent (LangGraph): goal -> priced subtasks -> hire workers -> run -> firewall -> judge -> settle.

Graph:  plan -> hire -> execute -> settle -> END
  plan     LLM (or mock) breaks the goal into 3-5 subtasks with capability, budget, deps, criteria.
  hire     registry.search() picks the best agent per subtask (meaning + trust + price).
  execute  runs subtasks in dependency order; every output goes through the firewall, then the judge.
           Quarantined/failed outputs are NOT passed to later subtasks.
  settle   builds the payout map (agent -> amount) for the backend to release from escrow.
"""
from __future__ import annotations

import uuid
from typing import TypedDict

from langgraph.graph import END, StateGraph

from . import config, llm
from .adapter import BaseWorker
from .firewall import get_firewall
from .judge import judge
from .mock_brain import topic_of
from .registry import AgentRegistry
from .schemas import FirewallVerdict, JudgeVerdict, Plan, RunResult, Subtask, WorkerResult

MAX_ATTEMPTS = 2  # original hire + one re-hire with the next-best agent

PLAN_SYSTEM = (
    "You are the master agent of an AI-agent marketplace. Break the user's goal into 3-5 subtasks that "
    "worker agents can do. Allowed capabilities: research, analysis, charts, deck, summary. "
    "Spread the total budget across subtasks by effort. acceptance_criteria: 2-3 short, concrete checks a "
    "small worker model can realistically meet in one reply (e.g. 'at least 3 bullet findings', 'names 2+ competitors'); "
    "do NOT demand citations, SWOT, or full reports. A 'charts' subtask produces exactly ONE chart; "
    "if you want two charts, make two subtasks. Return ONLY JSON: "
    '{"subtasks": [{"id": "t1", "title": str, "instructions": str, "capability": str, '
    '"depends_on": ["t0"], "budget": number, "acceptance_criteria": [str]}]}'
)


def mock_plan(goal: str, budget: float) -> dict:
    t = topic_of(goal)
    return {"subtasks": [
        {"id": "t1", "title": "Research", "instructions": f"Research {t}: market size, competitors, risks. Give bullet findings with sources.",
         "capability": "research", "depends_on": [], "budget": round(budget * 0.25, 2), "acceptance_criteria": ["at least 3 findings", "mentions competitors"]},
        {"id": "t2", "title": "Analysis", "instructions": f"Analyse the research on {t}: key insights and a chart-ready dataset of market share.",
         "capability": "analysis", "depends_on": ["t1"], "budget": round(budget * 0.25, 2), "acceptance_criteria": ["at least 3 insights", "chart data"]},
        {"id": "t3", "title": "Chart", "instructions": "Render a bar chart from the analysis dataset.",
         "capability": "charts", "depends_on": ["t2"], "budget": round(budget * 0.15, 2), "acceptance_criteria": ["PNG artifact"]},
        {"id": "t4", "title": "Deck", "instructions": f"Build a 5-slide deck on {t} from the research, analysis and chart.",
         "capability": "deck", "depends_on": ["t1", "t2", "t3"], "budget": round(budget * 0.25, 2), "acceptance_criteria": ["PPTX artifact", "at least 4 slides"]},
        {"id": "t5", "title": "Summary", "instructions": f"Write an executive summary of everything on {t}.",
         "capability": "summary", "depends_on": ["t1", "t2"], "budget": round(budget * 0.10, 2), "acceptance_criteria": ["under 150 words"]},
    ]}


class State(TypedDict, total=False):
    run_id: str
    goal: str
    budget: float
    plan: Plan
    workers: dict[str, BaseWorker]       # agent_id -> worker
    results: dict[str, WorkerResult]     # subtask_id -> result
    firewall: dict[str, FirewallVerdict]
    verdicts: dict[str, JudgeVerdict]
    final_output: str
    payouts: dict[str, float]
    attempts: list[dict]
    log: list[str]


class MasterAgent:
    def __init__(self, registry: AgentRegistry, workers: dict[str, BaseWorker]):
        self.registry, self.workers = registry, workers
        g = StateGraph(State)
        g.add_node("plan", self.plan)
        g.add_node("hire", self.hire)
        g.add_node("execute", self.execute)
        g.add_node("settle", self.settle)
        g.set_entry_point("plan")
        g.add_edge("plan", "hire"); g.add_edge("hire", "execute"); g.add_edge("execute", "settle"); g.add_edge("settle", END)
        self.graph = g.compile()

    # ------------------------------------------------------------------ nodes
    def plan(self, s: State) -> State:
        data, src = llm.chat_json(
            [{"role": "system", "content": PLAN_SYSTEM},
             {"role": "user", "content": f"Goal: {s['goal']}\nTotal budget: {s['budget']} credits"}],
            model=config.MASTER_MODEL, mock=lambda: mock_plan(s["goal"], s["budget"]))
        subtasks = []
        for i, d in enumerate((data or {}).get("subtasks", []), 1):
            d.setdefault("id", f"t{i}")
            if d.get("capability") not in ("research", "analysis", "charts", "deck", "summary"):
                d["capability"] = "research"
            subtasks.append(Subtask(**d))
        if not subtasks:
            subtasks = [Subtask(**d) for d in mock_plan(s["goal"], s["budget"])["subtasks"]]; src = "mock"
        scale = s["budget"] / max(sum(t.budget for t in subtasks), 1e-9)  # never exceed the buyer's budget
        for t in subtasks:
            t.budget = round(t.budget * min(scale, 1.0), 2)
        return {"plan": Plan(goal=s["goal"], total_budget=s["budget"], subtasks=subtasks, source=src),
                "log": s.get("log", []) + [f"planned {len(subtasks)} subtasks via {src}"]}

    def hire(self, s: State) -> State:
        log = []
        for t in s["plan"].subtasks:
            ranked = self.registry.search(f"{t.title}: {t.instructions}", top_k=1, capability=t.capability, max_price=t.budget or None)
            if not ranked:  # nothing within budget -> cheapest with the capability
                ranked = self.registry.search(t.instructions, top_k=1, capability=t.capability)
            t.assigned_agent = ranked[0]["agent"]["agent_id"] if ranked else None
            log.append(f"{t.id} -> {t.assigned_agent} (score {ranked[0]['score'] if ranked else 'n/a'})")
        return {"log": s["log"] + log}

    def execute(self, s: State) -> State:
        fw = get_firewall()
        results, fws, verdicts, log, attempts = {}, {}, {}, [], []
        done, pending = set(), list(s["plan"].subtasks)
        while pending:
            ready = [t for t in pending if set(t.depends_on) <= done] or pending[:1]  # break cycles
            for t in ready:
                pending.remove(t); done.add(t.id)
                # only clean + passed upstream results are visible to this worker
                ctx = {d: results[d] for d in t.depends_on if d in results and not fws[d].quarantined and verdicts[d].passed}
                tried: list[str] = []
                for attempt in range(MAX_ATTEMPTS):
                    worker = self.workers.get(t.assigned_agent or "")
                    if worker is None:
                        results[t.id] = WorkerResult(subtask_id=t.id, agent_id="none", output="[agent error] no agent available", source="error")
                    else:
                        results[t.id] = worker.run(t, ctx)
                    tried.append(results[t.id].agent_id)
                    fws[t.id] = fw.scan(results[t.id])
                    verdicts[t.id] = judge(t, results[t.id], fws[t.id])
                    log.append(f"{t.id} attempt {attempt + 1} ({results[t.id].agent_id}): fw={fws[t.id].severity} "
                               f"score={verdicts[t.id].final_score} -> {verdicts[t.id].escrow_action}")
                    attempts.append({"subtask_id": t.id, "attempt": attempt + 1, "agent_id": results[t.id].agent_id,
                                     "firewall": fws[t.id].severity, "score": verdicts[t.id].final_score,
                                     "escrow_action": verdicts[t.id].escrow_action,
                                     "reason": (verdicts[t.id].tier1_failures or [v.reason for v in verdicts[t.id].votes][:1] or [""])[0]})
                    if verdicts[t.id].passed:
                        break
                    # failed or quarantined: the agent gets nothing and loses trust; re-hire the next-best agent once
                    bad = results[t.id].agent_id
                    self.registry.trust[bad] = round(max(0.0, self.registry.trust.get(bad, 0.7) - (0.3 if fws[t.id].severity == "malignant" else 0.1)), 3)
                    ranked = [r for r in self.registry.search(f"{t.title}: {t.instructions}", top_k=5, capability=t.capability)
                              if r["agent"]["agent_id"] not in tried]
                    if not ranked:
                        break
                    t.assigned_agent = ranked[0]["agent"]["agent_id"]
                    log.append(f"{t.id}: re-hiring {t.assigned_agent}")
        return {"results": results, "firewall": fws, "verdicts": verdicts, "attempts": attempts, "log": s["log"] + log}

    def settle(self, s: State) -> State:
        payouts: dict[str, float] = {}
        for t in s["plan"].subtasks:
            v = s["verdicts"][t.id]
            if v.escrow_action == "release":
                aid = s["results"][t.id].agent_id
                payouts[aid] = round(payouts.get(aid, 0) + t.budget, 2)
            if v.passed:  # failures already lost trust during execute
                aid = s["results"][t.id].agent_id
                self.registry.trust[aid] = round(min(1.0, self.registry.trust.get(aid, 0.7) + 0.02), 3)
        final = next((s["results"][t.id].output for t in reversed(s["plan"].subtasks)
                      if t.capability == "summary" and s["verdicts"][t.id].passed), None)
        if final is None:
            final = "\n\n".join(f"## {t.title}\n{s['results'][t.id].output}" for t in s["plan"].subtasks if s["verdicts"][t.id].passed) or "No subtask passed."
        return {"payouts": payouts, "final_output": final, "log": s["log"] + [f"payouts {payouts}"]}

    # ------------------------------------------------------------------ public
    def run(self, goal: str, budget: float = 20.0) -> RunResult:
        s = self.graph.invoke({"run_id": uuid.uuid4().hex[:10], "goal": goal, "budget": budget, "log": []})
        order = [t.id for t in s["plan"].subtasks]
        return RunResult(run_id=s["run_id"], plan=s["plan"], results=[s["results"][i] for i in order],
                         firewall=[s["firewall"][i] for i in order], verdicts=[s["verdicts"][i] for i in order],
                         final_output=s["final_output"], payouts=s["payouts"], attempts=s.get("attempts", []))
