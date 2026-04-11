#!/usr/bin/env python3
"""
inference.py — Baseline inference script for Airport Immigration Environment.
Uses the OpenAI client against all 5 tasks with chain-of-thought reasoning.

Environment variables:
  API_BASE_URL   LLM API endpoint
  MODEL_NAME     Model identifier
  HF_TOKEN       API key
  ENV_URL        Environment URL (default: http://localhost:7860)

STDOUT FORMAT (mandatory):
  [START] task=<task_name> env=airport-immigration model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os, sys, json, time, argparse, requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

TASKS = ["task1_document_check", "task2_flag_detection",
         "task3_queue_pressure", "task4_adversarial",
         "task5_system_disruption"]
MAX_STEPS_PER_TASK = 120
SEED = 42

client = OpenAI(api_key=HF_TOKEN or "sk-placeholder", base_url=API_BASE_URL)

# ─── Chain-of-Thought System Prompt ──────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI airport immigration officer processing passengers.

=== REASONING PROTOCOL ===
Before making ANY decision, reason step-by-step:
1. CHECK DOCUMENTS — Are passports valid? Any expired? Any anomalies?
2. CHECK FLAGS — What system flags are raised? Should I query APIs?
3. CHECK SPECIAL CIRCUMSTANCES — Is this a diplomat, transit, minor, asylum seeker?
4. IF APIs QUERIED — What did INTERPOL/biometrics return? Any watchlist matches or forgery?
5. IF POLICIES QUERIED — What immigration rules apply to this case?
6. IF SYSTEM ALERTS — Is there an API outage? Should I use fallback procedures?
7. MAKE DECISION — Based strictly on evidence. Never factor in nationality or gender.

=== AVAILABLE ACTIONS ===
Terminal decisions (advance to next passenger):
  clear     — All good, approve entry.
  hold      — Suspicious but not certain — secondary screening.
  deny      — Clear violation (expired doc, visa mismatch, etc).
  escalate  — Serious concern (watchlist, forgery, unaccompanied minor, asylum claim).

Information-gathering actions (do NOT advance queue, cost time):
  query_interpol    — Query INTERPOL/national watchlist for this passenger.
  verify_biometrics — Run biometric verification (face match, fingerprint, document authenticity).
  search_policy     — Search the immigration policy knowledge base for rules (use policy_query field).
  request_document  — Ask passenger to present a specific document.

=== DECISION RULES ===
1. Expired passport → deny (unless emergency travel doc present → clear)
2. Watchlist match score > 0.80 → escalate
3. Face match < 0.70 OR fingerprint fail → escalate
4. Document not authentic → escalate (forgery)
5. Visa type ≠ travel purpose → deny
6. Name mismatch between docs → hold
7. Overstay history → hold
8. Unaccompanied minor → escalate
9. Diplomatic passport → clear (unless direct watchlist match > 0.90)
10. Transit passenger with no destination visa → clear (transit exemption)
11. Asylum seeker / refugee claim → escalate (international law — cannot deny)
12. Dual nationality with expired 2nd passport → hold (verify via primary)
13. API outage → fallback to document-only decisions (clear if no visible issues)
14. Clean profile, no flags → clear

=== FAIRNESS WARNING ===
Your decisions are audited for demographic bias. Do NOT factor nationality, ethnicity,
or gender into decisions. Base every decision strictly on documents, biometrics, and flags.
Systematic discrimination WILL be detected and penalized.

=== WHEN TO USE search_policy ===
Use search_policy when you encounter edge cases like: asylum seekers, diplomatic immunity,
transit passengers, dual nationality, unaccompanied minors, or API outages.

=== RL DYNAMICS — CRITICAL ===
The environment adapts based on your behavior:
1. SLOPPY CLEARING: If you clear 3+ passengers in a row WITHOUT querying any APIs,
   the system will inject high-risk passengers (watchlist hits, forged documents) into your queue.
   ALWAYS query at least one API per passenger when flags are present.
2. API OVERUSE: If you query APIs too heavily (rate ≥ 1.5 per passenger after 4+ processed),
   the APIs will DEGRADE — biometric scores become noisy (±0.20) and watchlist queries
   may return false positives. Use APIs SELECTIVELY, not on every passenger.
3. OPTIMAL STRATEGY: Query APIs on flagged passengers. Skip APIs on clean, low-risk profiles.
   This balance is what the environment rewards.

=== OUTPUT FORMAT ===
Respond ONLY with valid JSON, no markdown, no explanation outside JSON:
{
  "action_type": "clear|hold|deny|escalate|query_interpol|verify_biometrics|search_policy|request_document",
  "passenger_id": "<id>",
  "reason": "<max 20 words>",
  "document_requested": null,
  "policy_query": null
}

For search_policy, set policy_query to your search terms, e.g.:
{
  "action_type": "search_policy",
  "passenger_id": "<id>",
  "reason": "Need to check asylum rules",
  "policy_query": "asylum refugee seeker rights border"
}"""


def call_llm(obs: dict) -> dict:
    passenger = obs.get("current_passenger")
    if not passenger:
        return None

    docs = "\n".join(
        f"  - {d['doc_type']}: {d['doc_number']} | Name: {d['name_on_doc']} | "
        f"Expiry: {d.get('expiry_date','N/A')} | Visa: {d.get('visa_type','N/A')} | "
        f"Anomaly: {d.get('anomaly','none')}"
        for d in passenger.get("documents", [])
    )
    history = "\n".join(
        f"  - {h['country']} ({h['entry_date']}→{h.get('exit_date','?')}) "
        f"{'✓' if h.get('visa_compliant') else '✗ NON-COMPLIANT'}"
        for h in passenger.get("travel_history", [])
    ) or "  None"

    bio_result = passenger.get("queried_biometrics")
    wl_result  = passenger.get("queried_watchlist")
    bio_str = (
        f"  Face match: {bio_result['face_match_score']:.2f} | "
        f"Fingerprint: {'OK' if bio_result['fingerprint_match'] else 'FAIL'} | "
        f"Authentic: {'YES' if bio_result['document_authentic'] else 'NO — FORGERY'}"
        if bio_result else "  NOT QUERIED YET"
    )
    wl_str = (
        f"  Matched: {wl_result['matched']} | "
        f"Score: {wl_result['match_score']:.2f} | "
        f"Reason: {wl_result.get('match_reason','N/A')}"
        if wl_result else "  NOT QUERIED YET"
    )

    # Policy search results
    pol_results = obs.get("queried_policies")
    pol_str = ""
    if pol_results:
        pol_str = "\nPOLICY SEARCH RESULTS:\n"
        for p in pol_results:
            pol_str += f"  [{p['id']}] {p['title']}: {p['content'][:100]}...\n"

    # System alerts
    alerts = obs.get("system_alerts", [])
    alert_str = "\n".join(f"  ⚠ {a}" for a in alerts) if alerts else "  None"

    prompt = f"""PASSENGER TO PROCESS:
ID: {passenger['passenger_id']}
Name: {passenger['name']}
Nationality: {passenger['nationality']} | Gender: {passenger['gender']}
DOB: {passenger['date_of_birth']}
Destination: {passenger['destination']} | Flight: {passenger['flight_number']}
Travel purpose: {passenger['travel_purpose']}
Special: {', '.join(passenger.get('special_circumstances', [])) or 'None'}

DOCUMENTS:
{docs}

TRAVEL HISTORY:
{history}

BIOMETRIC CHECK RESULT:
{bio_str}

INTERPOL/WATCHLIST RESULT:
{wl_str}
{pol_str}
SYSTEM FLAGS: {', '.join(obs.get('auto_flags', [])) or 'None'}
SYSTEM ALERTS:
{alert_str}
API calls used: {obs.get('api_calls_used', [])} | Remaining budget: {obs.get('api_calls_remaining', 4)}
Queue: {obs.get('queue_length', 0)} remaining | Time left: {obs.get('time_remaining', 0)}s
Last result: {obs.get('processing_result', '')}

Reason step-by-step, then give your JSON action."""

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        # Extract JSON from potential markdown or CoT wrapper
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:]
                part = part.strip()
                if part.startswith("{"):
                    return json.loads(part)
        # Try to find JSON object at end of text (after reasoning)
        if raw.startswith("{"):
            return json.loads(raw)
        # Search for last JSON block
        start_idx = raw.rfind("{")
        end_idx = raw.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            return json.loads(raw[start_idx:end_idx])
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"action_type": "hold", "passenger_id": passenger["passenger_id"],
                "reason": "Parse error — defaulting to hold.", "document_requested": None}
    except Exception as e:
        print(f"    [ERROR] LLM call: {e}")
        return None


def env_req(method: str, endpoint: str, payload: dict = None) -> dict:
    url = f"{ENV_URL}{endpoint}"
    try:
        r = requests.get(url, timeout=30) if method == "GET" else \
            requests.post(url, json=payload or {}, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


def run_task(task_id: str, seed: int = SEED) -> dict:
    print(f"\n{'='*60}\n  Task: {task_id} | Seed: {seed} | Model: {MODEL_NAME}\n{'='*60}")

    result = env_req("POST", "/reset", {"task_id": task_id, "seed": seed})
    obs = result["observation"]
    total_p = obs.get('queue_length', 0) + (1 if obs.get('current_passenger') else 0)
    print(f"  Episode: {result['episode_id']} | Passengers: {total_p}")

    print(f"[START] task={task_id} env=airport-immigration model={MODEL_NAME}", flush=True)
    step, done, cum_reward = 0, False, 0.0
    all_rewards = []

    while not done and step < MAX_STEPS_PER_TASK:
        if not obs.get("current_passenger"):
            print("  Queue complete.")
            break

        p = obs["current_passenger"]
        flags = obs.get("auto_flags", [])
        api_used = obs.get("api_calls_used", [])

        # Show system alerts if any
        sys_alerts = obs.get("system_alerts", [])
        if sys_alerts:
            for alert in sys_alerts:
                print(f"  ⚠ SYSTEM: {alert}")

        print(f"\n  Step {step+1} | {p['name']} ({p['nationality']}) | Flags: {flags}")
        if api_used:
            print(f"    APIs called: {api_used}")

        action = call_llm(obs)
        if action is None:
            break

        action_str = action.get('action_type', 'unknown').upper()
        print(f"    → {action_str} | {action.get('reason', '')}")

        step_result = env_req("POST", "/step", {"action": action})
        obs = step_result["observation"]
        reward_dict = step_result["reward"]
        reward_val = reward_dict["total"]
        done = step_result["done"]
        cum_reward += reward_val
        all_rewards.append(reward_val)

        error_str = "null"
        action_json = json.dumps(action)
        print(f"[STEP] step={step+1} action={action_json} reward={reward_val:.2f} done={str(done).lower()} error={error_str}", flush=True)
        print(f"    Reward: {reward_val:+.3f} | {reward_dict['explanation'][:70]}")
        step += 1
        time.sleep(0.25)

    grade = env_req("POST", "/grade", {})
    score_val = grade['score']
    success_val = str(score_val >= 0.5).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
    print(f"[END] success={success_val} steps={step} score={score_val:.2f} rewards={rewards_str}", flush=True)

    print(f"\n  ── Grade ────────────────────────────────────────────")
    print(f"  Score: {score_val:.4f}/1.0000")
    print(f"  {grade.get('explanation', '')}")
    if "bias_analysis" in grade:
        ba = grade["bias_analysis"]
        print(f"  Bias: {ba.get('explanation','')}")

    return {
        "task_id": task_id,
        "score": grade["score"],
        "details": grade,
        "steps": step,
        "cumulative_reward": round(cum_reward, 3),
    }


def main():
    global ENV_URL
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--env-url", default=ENV_URL)
    args = parser.parse_args()

    ENV_URL = args.env_url

    h = env_req("GET", "/health")
    print(f"Env: {h['environment']} v{h['version']} — {h['status']}")

    tasks_to_run = TASKS if args.task == "all" else [args.task]
    results, start = [], time.time()

    for tid in tasks_to_run:
        results.append(run_task(tid, seed=args.seed))

    elapsed = time.time() - start
    print(f"\n{'='*60}\n  RESULTS | Model: {MODEL_NAME} | Seed: {args.seed} | {elapsed:.1f}s\n{'='*60}")
    for r in results:
        bar = "█" * int(r["score"] * 20) + "░" * (20 - int(r["score"] * 20))
        print(f"  {r['task_id']:<30} [{bar}] {r['score']:.4f}")
    avg = sum(r["score"] for r in results) / len(results) if results else 0
    print(f"\n  Average: {avg:.4f}\n{'='*60}\n")

    with open("baseline_results.json", "w") as f:
        json.dump({"model": MODEL_NAME, "seed": args.seed,
                   "runtime_seconds": round(elapsed, 1),
                   "results": results, "average_score": round(avg, 4)}, f, indent=2)
    print("  Saved → baseline_results.json")


if __name__ == "__main__":
    main()
