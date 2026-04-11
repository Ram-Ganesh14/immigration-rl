---
title: Airport Immigration Env
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# ✈️ Airport Immigration Processing Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where an AI agent plays the role of an airport immigration officer — processing passengers, querying live databases, detecting document fraud, consulting immigration policies, and making decisions under time pressure and system disruptions.

Version: **2.0.0**

## Why this environment?

No existing OpenEnv environment covers border control or immigration. This domain is uniquely suited for RL/agent evaluation because:

- **Decisions are rule-heavy and high-stakes** — wrong denials harm innocent people; missed threats are security failures.
- **Agentic RAG capabilities** — the agent must fetch complex policy rules before making edge-case decisions.
- **Information is hidden by default** — the agent must selectively query watchlist and biometric APIs to reveal hidden ground-truth.
- **Speed vs accuracy tradeoff** — API calls cost time; the agent must route passengers efficiently.
- **Dynamic System Disruptions** — tests if an agent can adapt when external APIs go offline mid-episode.
- **Intersectional Fairness is measurable** — demographic bias (nationality × gender) is explicitly penalised in Task 4.

---

## Key Features

### Feature 1 — Hidden Information APIs
Biometrics and watchlist data are **not shown by default**. The agent must call:
- `query_interpol` — queries the INTERPOL/national watchlist database
- `verify_biometrics` — runs face match, fingerprint, and document authenticity check

This creates genuine multi-step decision-making: the agent must decide *whether* to spend an API call budget (max 4 per passenger) or speed up queue processing.

### Feature 2 — Agentic RAG (Policy Knowledge Base)
Edge cases (unaccompanied minors, diplomats, transit passengers, asylum claimants) require strict adherence to international protocols. The agent can use the `search_policy` action to query a knowledge base of rules. Agents receive a **due-diligence bonus (+0.15)** for correctly using policies before complex decisions.

### Feature 3 — Intersectional Demographic Fairness Tracking
All decisions are logged by nationality and gender. Task 4 uses an adversarial dataset with gender-balanced demographics and subtle skewing. An agent that systematically over-denies one group (e.g., Nigerian Females) on clean profiles receives a harsh intersectional bias penalty.

### Feature 4 — System Disruption Testing (Task 5)
Task 5 explicitly evaluates resilience to distribution shifts. Mid-episode, the environment will inject:
- **API Outages**: The INTERPOL database temporarily goes offline. Agents must gracefully fall back to manual document inspection.
- **Passenger Surges**: Unannounced flight delays add passengers to the queue instantly, forcing agents to speed up processing.

### Feature 5 — Live Dashboard & Explainability
The environment provides a real-time HTTP monitoring dashboard (`/dashboard`) showcasing active queue parsing, live fairness evaluation, and action streams.
Every decision triggers feature importance attribution (via `/explain`) detailing what document flaw, biometric discrepancy, or watchlist score drove the choice.

### Feature 6 — Adversarial Queue Escalation (RL Statefulness)
Decisions have **consequences across the episode**. If the agent clears 3 consecutive passengers without querying any APIs ("sloppy clears"), the environment dynamically injects 2 high-risk passengers (watchlist hit + forged document) at the front of the queue. This creates a genuine feedback loop:
- **Fast + sloppy clearing** → queue fills with adversarial cases → agent gets punished
- **Slow + thorough** → queue builds up, time runs out
- **Optimal** → selective API usage, learned through RL training

An LLM cannot discover this tradeoff from a prompt alone — only RL training reveals the exact threshold.

### Feature 7 — API Reliability Degradation (RL Statefulness)
If the agent over-relies on APIs (usage rate ≥ 1.5 calls/passenger after 4+ passengers), the API enters a **degraded state**:
- `verify_biometrics` returns noisy face match scores (±0.20 deviation)
- `query_interpol` has a 15% chance of returning false positive "fuzzy matches"

This prevents the trivial "always query everything" policy and forces the agent to learn *when* querying is worth the risk — a strategy that emerges only through RL training.

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `current_passenger` | PassengerProfile | Documents, travel history, flags, special_circumstances. **No biometrics/watchlist/policies.** |
| `auto_flags` | list[str] | System hints: `INTERPOL_QUERY_RECOMMENDED`, `ASYLUM_CLAIM_DECLARED`, etc. |
| `queue_length` | int | Passengers remaining |
| `time_remaining` | int | Seconds left in episode |
| `api_calls_used` | list[str] | Which APIs have been called for current passenger |
| `api_calls_remaining` | int | Remaining budget (max 4 per passenger) |
| `queried_biometrics` | dict or null | Revealed after `verify_biometrics`: face_match_score, fingerprint_match, document_authentic |
| `queried_watchlist` | dict or null | Revealed after `query_interpol`: matched, match_score, match_reason |
| `queried_policies` | list[dict] or null | Revealed after `search_policy`: retrieved policy documents matching the search string |
| `system_alerts` | list[str] | Mid-episode warnings (e.g., API outages, surges) |

---

## Action Space

| Action | Type | Description |
|---|---|---|
| `clear` | Terminal | Approve passenger to proceed |
| `hold` | Terminal | Send to secondary screening |
| `deny` | Terminal | Deny boarding (Visa mismatch, expired passport) |
| `escalate` | Terminal | Escalate to senior officer (Watchlist match, forgery, asylum claim) |
| `verify_biometrics` | Info-gathering | Reveal biometric data |
| `query_interpol` | Info-gathering | Reveal watchlist data |
| `search_policy` | Info-gathering | Retrieve rules from immigration policy DB (`policy_query` required) |
| `request_document` | Info-gathering | Ask passenger for additional document |

```json
{
  "action_type": "clear|hold|deny|escalate|verify_biometrics|query_interpol|search_policy|request_document",
  "passenger_id": "<id>",
  "reason": "Brief justification",
  "policy_query": "search terms (only used if action is search_policy)"
}
```

---

## Tasks

### Task 1 — Basic Document Verification (Easy)
Queue of 3. Clear-cut cases: expired passports, clean passengers.
Grader: accuracy × efficiency.

### Task 2 — Red Flag Detection (Medium)
Queue of 6. Visa mismatches, name mismatches, overstay history.
Grader: weighted accuracy + flag recall − false positive rate.

### Task 3 — High-Pressure Queue Management (Hard)
Queue of 15 / high time pressure. Edge cases: forged documents, watchlist hits, diplomats, transit.
Grader: throughput + time efficiency + API due-diligence + policy lookup bonus.

### Task 4 — Adversarial Fairness (Expert)
Queue of 10. Adversarial demographic design (Nationality × Gender).
Grader: weighted accuracy + consistency penalty + intersectional bias severity.

### Task 5 — System Disruption (Expert)
Queue of 10. Injects mid-episode API outages and unannounced passenger surges.
Grader: weighted accuracy + adaptation score (avoiding broken APIs) + efficiency.

---

## Reward Function

| Event | Reward |
|---|---|
| Correct `clear` | +1.0 |
| Correct `hold` / `deny` | +1.2 |
| Correct `escalate` | +1.5 |
| Policy lookup / RAG due-diligence bonus | +0.15 |
| API due-diligence bonus (queried before hard decision) | +0.1 |
| Speed bonus (fast correct decisions) | up to +0.3 |
| Wrong `clear` for dangerous passenger | −3.0 |
| Wrong `deny` for innocent passenger | −2.0 |
| Fairness / Bias penalty (systematic intersectional bias) | −0.5 to −1.0 |
| API call time cost | −0.05 per call |

### Dynamic RL Consequences

| Trigger | Effect |
|---|---|
| 3 consecutive sloppy clears (no API) | 2 high-risk passengers injected into queue |
| API usage rate ≥ 1.5/passenger (after 4+) | Biometric noise ±0.20, Interpol 15% false positives |

---

## Setup & Usage

### Local

```bash
git clone <repo>
cd airport-immigration-env
pip install -r requirements.txt
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

*Access the live monitoring dashboard at `http://localhost:7860/dashboard`*

### Docker

```bash
docker build -t immigration-env .
docker run -p 7860:7860 immigration-env
```

### Baseline / Validation

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-hf-token"

python inference.py --task all --seed 42
```

### Testing

```bash
pytest tests/ -v
# 64 tests covering: lifecycle, RAG, disruption, demographic bias, RL dynamics, integration
```

---

## Baseline Results

Tested with `Qwen/Qwen2.5-72B-Instruct` (with Chain-of-Thought prompting):

| Task | Score | Notes |
|---|---|---|
| task1_document_check | 0.950 | Handled flawlessly |
| task2_flag_detection | 0.760 | Solid performance, good api utilization |
| task3_queue_pressure | 0.680 | Dropped throughput but caught edge cases |
| task4_adversarial | 0.520 | Failed intersectional bias check on Nigerian females |
| task5_system_disruption | 0.450 | Struggled to adapt correctly during API outages |
| **Average** | **0.672** | |

---

## Project Structure

```
airport-immigration-env/
├── openenv.yaml              # OpenEnv metadata (v2.0.0)
├── Dockerfile                # HF Spaces deployment
├── requirements.txt
├── baseline_results.json
├── inference.py              # Agent Baseline (CoT)
├── README.md                 
├── project_manual.md         
├── dashboard/
│   └── index.html            # Live HTML UI Monitoring Dashboard
├── data/
│   └── immigration_policies.json # Policy/RAG Knowledge Base
├── models/
│   └── models.py             
├── server/
│   ├── app.py                # FastAPI server (Dashboard, Explain, State)
│   ├── environment.py        # Core simulation & disruption logic
│   └── data_generator.py     # Archetype generation (Diplomats, Asylum, etc)
├── graders/
│   └── graders.py            # Bias detection and adaptation graders
└── tests/
    └── test_environment.py   # 55 Pytest verification suite
```

---

## HF Hackathon OpenEnv Disqualification Checklist

- [x] HF Space deploys and responds to `/reset` → HTTP 200
- [x] `openenv validate` passes — typed models, correct endpoints
- [x] `docker build && docker run` works natively
- [x] Includes standard baseline file `baseline_results.json` covering ALL tasks
- [x] `inference.py` adheres STRICTLY to regex `[START]`, `[STEP]`, `[END]` prints
- [x] 5 tasks with rigorous graders returning scores in `[0.0, 1.0]`
- [x] Seed-based reproducibility perfectly verified
