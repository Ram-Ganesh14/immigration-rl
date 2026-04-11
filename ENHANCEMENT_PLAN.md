# 🚀 Airport Immigration Env — Enhancement Plan (Standalone)

> **Context:** Meta PyTorch OpenEnv Hackathon (Scaler SST)
> **Project:** Airport Immigration Processing Environment
> **Submission:** #7 (Validated — Phase 1 & 2 passed)
> **Deadline:** 12th April 2026, 11:59 PM IST

---

## Scoring Rubric (from hackathon)

| Weight | Criteria | Description |
|---|---|---|
| **30%** | Real-world utility | Genuine task? Would someone use this to train/evaluate agents? |
| **25%** | Task & grader quality | Well-defined tasks? Fair graders? Difficulty progression? |
| **20%** | Environment design | Clean state management? Good reward shaping? Sensible action/observation? |
| **15%** | Code quality & spec | OpenEnv spec, Docker, typed models, tested |
| **10%** | Creativity & novelty | Novel domain? Interesting mechanics? Clever reward? |

---

## Current State (Pre-Enhancement)

**What's already working:**
- 4 tasks (easy → expert) with graders scoring 0.0–1.0
- Hidden information APIs (query_interpol, verify_biometrics) — agent must proactively query
- Shaped rewards: speed bonus, API due-diligence bonus, fairness penalty
- Demographic bias tracking (nationality) in Task 4
- 32 passing pytest tests
- Full OpenEnv compliance, Docker, HF Spaces deployment
- Baseline inference.py with LLM

**Gaps identified:**
- No RAG/knowledge-retrieval action
- Only 9 passenger archetypes (missing diplomatic, transit, asylum, dual-nationality)
- Bias tracking only by nationality (not gender or intersectional)
- No dynamic/stochastic events mid-episode
- No explainability features
- No visual dashboard
- baseline_results.json incomplete (only Task 1)

---

## Enhancements (7 total + 3 bug fixes)

### 🔧 Bug Fixes (15 min)

1. **baseline_results.json** — Only shows Task 1 score. Must include all tasks.
2. **project_manual.md** — References `server/main.py` but file is `server/app.py`.
3. **README version** — Says v1.0.0 but openenv.yaml says v2.0.0.

---

### Enhancement 1: `search_policy` RAG Action ⭐⭐⭐

**Impact:** Real-world utility (+5%), Creativity (+5%), Environment design (+3%)

**What:** Add a new info-gathering action `search_policy` that lets the agent search an immigration policy knowledge base before making edge-case decisions.

**Files:**
- `[NEW] data/immigration_policies.json` — 15 immigration rules with tags
- `[MODIFY] models/models.py` — Add `SEARCH_POLICY` to ActionType, `policy_query` to Action, `queried_policies` to Observation
- `[MODIFY] server/environment.py` — Add `_handle_search_policy()` handler, policy bonus in reward
- `[MODIFY] inference.py` — Update system prompt with search_policy awareness

**How it works:**
```
Agent sees: "UNACCOMPANIED_MINOR" flag
Agent action: search_policy(query="unaccompanied minor rules")
Environment returns: POL-002 "Passengers under 16 without guardian must be escalated"
Agent action: escalate (correct!)
Bonus: +0.15 for using policy before deciding
```

---

### Enhancement 2: 4 New Passenger Archetypes ⭐⭐

**Impact:** Real-world utility (+3%), Task quality (+2%)

**Files:**
- `[MODIFY] server/data_generator.py` — Add 4 new generator methods

| Method | Scenario | Ground Truth | Tests |
|---|---|---|---|
| `diplomatic_passport()` | Diplomatic passport, minor flags | `clear` | Diplomatic immunity respect |
| `transit_passenger()` | No destination visa, connecting flight | `clear` | Transit rule understanding |
| `dual_nationality()` | Two passports, one expired | `hold` | Multi-document verification |
| `refugee_claimant()` | No visa, claims asylum | `escalate` | Asylum seeker rights |

---

### Enhancement 3: Intersectional Bias Detection ⭐⭐

**Impact:** Task quality (+2%), Creativity (+2%)

**Files:**
- `[MODIFY] graders/graders.py` — Extend `_demographic_bias_penalty()` to check gender bias and intersectional (nationality × gender) bias
- `[MODIFY] server/data_generator.py` — Balance gender in adversarial queue

---

### Enhancement 4: Chain-of-Thought Inference ⭐⭐

**Impact:** Code quality (+3%), Environment design (+2%)

**Files:**
- `[MODIFY] inference.py` — Add step-by-step reasoning prompt, verify [START]/[STEP]/[END] log format

**System prompt addition:**
```
Before deciding, reason step by step:
1. Check documents — expired? mismatches?
2. Check flags — should I query APIs?
3. Evaluate API results if available
4. Check special circumstances (ETD, minor, diplomatic, asylum)
5. If unsure, search_policy for clarification
6. Make decision based strictly on evidence
```

---

### Enhancement 5: Task 5 — System Disruption ⭐⭐⭐

**Impact:** Creativity (+5%), Task quality (+3%)

**What:** A 5th task where mid-episode crises occur: API outages, passenger surges, security alerts. Tests agent robustness under distribution shift.

**Files:**
- `[MODIFY] server/environment.py` — Add task5 config, crisis injection logic in step()
- `[MODIFY] models/models.py` — Add `system_alerts` to Observation
- `[NEW in] graders/graders.py` — Add `grade_task5()` with adaptation scoring
- `[MODIFY] openenv.yaml` — Add task5 entry

**Crisis timeline:**
- Step 4: INTERPOL_API_OFFLINE (agent must fall back to document-only)
- Step 7: SURGE — 3 new passengers added to queue
- Step 9: INTERPOL_API_RESTORED

---

### Enhancement 6: Explainability Endpoint ⭐⭐

**Impact:** Real-world utility (+2%), Creativity (+2%)

**Files:**
- `[MODIFY] server/app.py` — Add `GET /explain` endpoint

**Returns:**
```json
{
  "passenger_id": "abc123",
  "decision": "deny",
  "feature_importance": {
    "passport_validity": 0.45,
    "visa_match": 0.30,
    "watchlist_score": 0.05
  },
  "key_factors": ["Passport expired 45 days ago"],
  "confidence": 0.92
}
```

---

### Enhancement 7: Live Monitoring Dashboard ⭐⭐⭐⭐

**Impact:** Creativity (+3%), Real-world utility (+2%)

**Files:**
- `[NEW] dashboard/index.html` — Single-page HTML/CSS/JS app
- `[MODIFY] server/app.py` — Mount static files, add /dashboard route

**Features:**
- Passenger queue visualization (cards)
- Real-time decision feed with ✓/✗
- Fairness gauge (donut chart by nationality)
- API call tracker with animations
- Cumulative reward graph

---

## Verification Checklist

- [ ] `pytest tests/ -v` — all tests pass (40+ tests)
- [ ] `python -m uvicorn server.app:app` — starts, `/health` returns 200
- [ ] All 5 tasks grade correctly via `/grade`
- [ ] `docker build && docker run` works
- [ ] `python inference.py --task all --seed 42` completes < 20 min
- [ ] `[START]`/`[STEP]`/`[END]` log format matches hackathon spec
- [ ] Graders never return constant score
- [ ] Push to GitHub + HuggingFace Spaces
- [ ] HF Space `/reset` returns 200
