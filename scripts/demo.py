"""Console demo for the pitch: run a goal, watch the shady agent get caught, audit the lazy one."""
import json

from ai_layer import build_marketplace
from ai_layer.audit import audit_agent

registry, workers, master = build_marketplace()
goal = "Make me a market analysis deck on AI agent marketplaces in India"
print(f"\nGOAL: {goal}\n")
res = master.run(goal, budget=20)
caps = {t.id: t.capability for t in res.plan.subtasks}
print(f"{'id':3} {'try':3} {'capability':9} {'agent':20} {'firewall':10} {'judge':6} action")
for a in res.attempts:
    print(f"{a['subtask_id']:3} {a['attempt']:<3} {caps[a['subtask_id']]:9} {a['agent_id']:20} {a['firewall']:10} {a['score']:<6} {a['escrow_action']}"
          + (f"   <- {a['reason'][:70]}" if a['escrow_action'] != 'release' and a['reason'] else ""))
print("\nPAYOUTS:", json.dumps(res.payouts))
print("\nFINAL OUTPUT:\n", res.final_output[:600])
print("\nSECRET-SHOPPER AUDITS")
for aid in ["agent-analysis-01", "agent-lazy-01"]:
    a = audit_agent(workers[aid], n_tasks=6, seed=3)
    print(f"  {aid:20} claimed {a.claimed_quality:.2f}  audited {a.audited_accuracy:.2f}  trust {a.trust_score:.2f}  {'FLAGGED' if a.flagged else 'ok'}")
print("\nTRUST SCORES:", registry.trust)
