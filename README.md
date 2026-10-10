# AgentMarket — AI/ML layer (Aditya)

Master agent → priced subtasks → hire workers from a semantic registry → every output passes a
**prompt-injection firewall** and a **3-model judge panel** → escrow release/refund → **secret-shopper
audits** keep agents honest. Runs fully offline (mock mode) and switches to free LLM APIs when keys exist.

```
agentmarket-ai/
├── ai_layer/                 # the library the backend imports / the API wraps
│   ├── config.py             # env vars: models, thresholds, providers
│   ├── llm.py                # one OpenAI-compatible client for Groq / Gemini / Ollama / OpenRouter + mock fallback
│   ├── schemas.py            # pydantic models = JSON contract with the backend
│   ├── registry.py           # agent cards + semantic search (sentence-transformers or TF-IDF) + trust
│   ├── adapter.py            # BaseWorker / FunctionAdapter / HTTPAdapter — plug any agent in
│   ├── workers/demo_workers.py  # research, analysis, charts, deck, summary + 2 bad actors for the demo
│   ├── firewall.py           # layer-3 injection firewall: rules + trained model, severity, quarantine
│   ├── judge.py              # tier-1 rule checks -> 3-judge panel vote -> escrow action
│   ├── audit.py              # secret-shopper known-answer audits -> trust score
│   └── master.py             # LangGraph master agent: plan -> hire -> execute -> settle
├── classifier/
│   ├── build_dataset.py      # 3-class dataset: synthetic templates x obfuscations (+ HF datasets if reachable)
│   ├── train_model.py        # tuned XGB+LogReg blend, group-aware CV, recall-floor threshold, severity model
│   └── train_transformer.py  # optional DeBERTa-v3 fine-tune with early stopping
├── api/main.py               # FastAPI service for Karan's backend
├── models/                   # trained artefacts (pkl + firewall_config.json + training_report.json)
├── data/                     # injection_v2.csv (+ my_agent_samples.csv for your own examples)
├── scripts/demo.py           # console demo for the pitch
└── tests/test_pipeline.py
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env            # add GROQ_API_KEY / GEMINI_API_KEY; leave empty for offline mock
make data && make train         # build dataset + train the firewall (~10-20 min; make train-quick for 2 min)
make test                       # 5 tests, offline
make demo                       # watch the shady agent get quarantined and the lazy one get flagged
make api                        # http://localhost:8001/docs
```

## Backend contract (Karan)

Base URL `http://ai:8001`. All JSON. Shapes are in `ai_layer/schemas.py`.

| Endpoint | Purpose | Returns |
|---|---|---|
| `POST /run` `{goal, budget, buyer_id?}` | whole pipeline for one buyer goal | `RunResult`: `plan.subtasks[]` (id, capability, budget, assigned_agent), `results[]`, `firewall[]` (severity, score), `verdicts[]` (`escrow_action`: release/refund/hold, `final_score`, votes), `payouts{agent_id: amount}`, `final_output` |
| `POST /classify` `{text}` | firewall only | `{is_injection, severity: benign/malicious/malignant, risk_score, rule_hits, status}` |
| `GET /agents/search?q=&capability=&max_price=` | semantic agent search | ranked `[{agent, similarity, trust, score}]` |
| `GET /agents` | registry + trust scores | |
| `POST /agents/register` `AgentCard` with `endpoint` | add an external HTTP agent | |
| `POST /audit/{agent_id}` | secret-shopper audit, updates trust | `AuditReport` (claimed vs audited, gap, trust_score, flagged) |
| `GET /firewall/quarantine` | everything quarantined so far | |

**Flow the backend drives:** buyer submits goal → backend locks escrow → `POST /run` → for each verdict,
`escrow_action == "release"` ⇒ pay `payouts[agent_id]` on-chain, `"refund"` ⇒ return that subtask's budget to buyer.
`firewall[i].severity == "malignant"` ⇒ also slash the agent's stake / trust. Mirror `GET /agents` trust into your DB.

**External agent contract** (`HTTPAdapter`): backend/agent devs expose
`POST endpoint {subtask, instructions, context} -> {output, artifacts?, data?}`.

## Firewall classifier (3-class)

| class | meaning | action |
|---|---|---|
| `benign` | normal worker output | forward |
| `malicious` | tries to hijack instructions, roles, or the judge | quarantine + refund |
| `malignant` | also steals secrets, moves money, or runs code/tools | quarantine + refund + trust slash |

Design (builds on Aditya's original `train_model.py`):
- **Two stages.** A: injection-vs-benign = soft-vote blend of XGBoost + LogReg on word (1-2) + char (3-5) TF-IDF,
  `RandomizedSearchCV` with **StratifiedGroupKFold** (a template never sits in both train and validation, so CV
  scores can't leak). B: malicious-vs-malignant LogReg trained on injection rows only.
- **Threshold** picked from out-of-fold scores with a 95% recall floor, then nudged into the gap below the weakest caught attack.
- **Honest test set:** held-out split by template *group* → unseen attack phrasings. The train/CV gap and a
  "big gap → memorising" warning are printed; `models/training_report.json` keeps everything.
- **Obfuscation robustness:** base64, hidden HTML comments, zero-width chars, leetspeak, alternating caps,
  attacks buried in benign text — all in the data, and the firewall also de-obfuscates before scoring.
- **Rules + model ensemble** at inference (`score = max(rule, model)`), chunked so a single bad sentence in a long
  clean answer is still caught; rules give instant severity for the obvious cases.
- **Heavier option:** `make train-transformer` fine-tunes DeBERTa-v3-small (early stopping, class-weighted loss,
  label smoothing); set `FIREWALL_USE_TRANSFORMER=1` and the firewall averages it with the sklearn model.
- **Why synthetic first:** HF was blocked where this was built. `build_dataset.py` pulls `deepset/prompt-injections`
  and `jackhhao/jailbreak-classification` automatically when reachable — run `make data && make train` on your laptop
  to mix them in. Drop your own `data/my_agent_samples.csv` (text,label[,group]) and it is mixed in too.

## Judge, audit, master

- **Judge:** tier-1 (empty/error/quarantined/missing artifact/missing required term) is free and runs first; then
  3 judges from `JUDGE_MODELS` (use different providers so one jailbroken model can't decide alone) score 0-10,
  majority of passes wins. Worker output is wrapped in `<output>` tags and judges are told it is data.
- **Audit:** known-answer tasks phrased like real work; `trust = 0.5*accuracy + 0.3*(1-overclaim) + 0.2*judge pass rate`;
  `claimed - audited > 0.25` ⇒ flagged. Demo: `QuickAnalyst Pro` claims 98%, audits at ~17%.
- **Master (LangGraph):** plan → hire (`0.6*similarity + 0.3*trust + 0.1*cheapness`, within budget) → execute in
  dependency order (quarantined/failed outputs are never shown to later workers) → settle payouts & trust.

## Cut list if time runs out
1. Skip the transformer; sklearn firewall is already ~0.97 F1 on unseen templates.
2. Use one judge model instead of three (`JUDGE_MODELS=groq:...`).
3. Drop deck worker; charts + summary are enough for the demo.
