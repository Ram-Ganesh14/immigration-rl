---
title: Airport Immigration Processing Environment
emoji: 🛂
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
tags:
  - openenv
  - rl
  - immigration
  - airport
  - document-verification
  - queue-management
  - decision-making
---

# ✈️ Airport Immigration Processing Environment

An [OpenEnv](https://github.com/openenv/openenv) environment where an AI agent plays the role of an airport immigration officer, processing departing passengers under time pressure with real-world decision rules.

## Why this environment?

No existing OpenEnv environment covers border control or immigration processing — a domain where:
- Decisions are **rule-heavy and high-stakes** (wrong denials harm innocent people; missed threats are security failures)
- **Partial information** is the norm (biometric confidence scores, fuzzy watchlist matches, translated documents)
- **Time pressure** creates a throughput vs accuracy tradeoff
- **Consistency/fairness** across passengers of different backgrounds is explicitly measurable

This makes it an ideal benchmark for evaluating LLM decision-making under structured constraints.

---

## Observation Space

At each step the agent receives:

| Field | Type | Description |
|---|---|---|
| `current_passenger` | PassengerProfile | Full profile of the passenger being processed |
| `queue_length` | int | Number of passengers still waiting |
| `queue_summary` | list | Brief info on the next 5 passengers |
| `time_remaining` | int | Seconds left in the episode |
| `auto_flags` | list[str] | System-detected anomalies (e.g. `PASSPORT_EXPIRED`, `WATCHLIST_MATCH`) |
| `processing_result` | str | Feedback from the last action |
| `fairness_score` | float | Drops if agent makes inconsistent decisions for similar profiles |

**PassengerProfile includes:**
- Basic info (name, nationality, DOB, destination, purpose)
- Documents: passport, visa, boarding pass, optional emergency travel doc
- Biometrics: face match score (0.0–1.0), fingerprint match
- Watchlist: fuzzy match score + reason
- Travel history: prior entries with visa compliance flags
- Special circumstances: unaccompanied minor, medical emergency, etc.
- Auto-detected flags (shown to agent)

---

## Action Space

| Action | When to use |
|---|---|
| `clear` | All documents valid, no flags, approve to proceed |
| `hold` | Suspicious but not definitive — secondary screening |
| `deny` | Clear violation (expired passport, visa mismatch) |
| `escalate` | Watchlist match, biometric failure, unaccompanied minor, document forgery suspected |
| `request_document` | Ask passenger for additional document (max 2 per passenger) |

```json
{
  "action_type": "clear|hold|deny|escalate|request_document",
  "passenger_id": "<id from observation>",
  "reason": "Brief justification",
  "document_requested": null
}
```

---

## Tasks

### Task 1 — Basic Document Verification (Easy)
- **Queue**: 3 passengers
- **Issues**: Expired passports, straightforward mismatches
- **Grader**: Simple accuracy × efficiency
- **Expected baseline score**: ~0.70

### Task 2 — Red Flag Detection (Medium)
- **Queue**: 6 passengers
- **Issues**: Visa purpose mismatches, overstay history, name mismatches, watchlist hits
- **Grader**: Weighted accuracy (dangerous cases penalised more) + flag recall − false positive rate
- **Expected baseline score**: ~0.55

### Task 3 — High-Pressure Queue Management (Hard)
- **Queue**: 15 passengers
- **Issues**: All of the above + emergency travel docs overriding expired passports, biometric fraud, unaccompanied minors
- **Grader**: Weighted accuracy + throughput + time efficiency + edge case bonus
- **Expected baseline score**: ~0.40

### Task 4 — Adversarial Edge Cases (Expert)
- **Queue**: 8 passengers
- **Issues**: Forged documents with plausible biometrics, partial watchlist matches, contradictory profile information
- **Grader**: Weighted accuracy + consistency/fairness score − over-escalation penalty
- **Expected baseline score**: ~0.35

---

## Reward Function

| Event | Reward |
|---|---|
| Correct `clear` | +1.0 |
| Correct `hold` | +1.2 |
| Correct `deny` | +1.2 |
| Correct `escalate` | +1.5 |
| Wrong `clear` for dangerous passenger | −3.0 |
| Wrong `deny` for innocent passenger | −2.0 |
| Over-escalation | −0.3 |
| Other wrong decision | −1.5 |
| Speed bonus (fast correct decisions) | up to +0.3 |
| Fairness penalty (inconsistent decisions) | −0.5 |
| Excessive document requests (>3 per passenger) | −0.1/action |

---

## Setup & Usage

### Local

```bash
git clone https://github.com/Ram-Ganesh14/immigration-rl
cd immigration-rl
pip install -r requirements.txt
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t immigration-env .
docker run -p 7860:7860 immigration-env
```

### API usage

```bash
# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_document_check", "seed": 42}'

# Take action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "clear",
      "passenger_id": "<id>",
      "reason": "All documents valid"
    }
  }'

# Get state
curl http://localhost:7860/state

# Grade episode
curl -X POST http://localhost:7860/grade
```

### Run baseline inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
export ENV_URL="http://localhost:7860"

python inference.py --task all --seed 42
```

### Run tests

```bash
pytest tests/ -v
```

---

## Baseline Results

Tested with `gpt-4o-mini`, seed=42:

| Task | Score | Notes |
|---|---|---|
| task1_document_check | 0.72 | Misses edge cases occasionally |
| task2_flag_detection | 0.54 | Struggles with subtle visa mismatches |
| task3_queue_pressure | 0.41 | Time pressure causes rushed decisions |
| task4_adversarial | 0.33 | Inconsistency penalised by fairness metric |
| **Average** | **0.50** | |

---

## Project Structure

```
airport-immigration-env/
├── openenv.yaml              # OpenEnv metadata
├── Dockerfile                # Containerised deployment
├── requirements.txt
├── inference.py              # Baseline inference script (required)
├── README.md
├── __init__.py               # Package root exports
├── client.py                 # ImmigrationEnv HTTP client (sync + async)
├── pyproject.toml            # Python package definition
├── models/
│   └── models.py             # Pydantic: Observation, Action, State, Reward
├── server/
│   ├── app.py                # FastAPI server
│   ├── environment.py        # ImmigrationEnvironment core logic
│   └── data_generator.py     # Synthetic passenger factory (seeded)
├── graders/
│   └── graders.py            # All 4 task graders (0.0–1.0)
├── outputs/                  # gitignored runtime outputs
│   ├── logs/
│   └── evals/
└── tests/
    └── test_environment.py   # 27 pytest tests
```

---

## Disqualification Checks

- [x] HF Space deploys and responds to `/reset` with HTTP 200
- [x] `openenv validate` passes (typed models, correct endpoints)
- [x] `docker build && docker run` works
- [x] `inference.py` runs end-to-end and produces scores
- [x] 4 tasks with graders returning 0.0–1.0
- [x] Seed-based reproducibility (same seed = same episode = same score)

---

## Environment variables required

```
API_BASE_URL    LLM API endpoint (e.g. https://api.openai.com/v1)
MODEL_NAME      Model identifier (e.g. gpt-4o-mini)
HF_TOKEN        Hugging Face / API key
ENV_URL         Environment URL (default: http://localhost:7860)
```
