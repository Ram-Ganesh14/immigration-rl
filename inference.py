#!/usr/bin/env python3
"""
inference.py — Baseline inference script.
Uses the OpenAI client against all 4 tasks.

Environment variables:
  API_BASE_URL   LLM API endpoint
  MODEL_NAME     Model identifier
  HF_TOKEN       API key
  ENV_URL        Environment URL (default: http://localhost:7860)
"""

import os, sys, json, time, argparse, requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

TASKS = ["task1_document_check", "task2_flag_detection",
         "task3_queue_pressure", "task4_adversarial"]
MAX_STEPS_PER_TASK = 120
SEED = 42

client = OpenAI(api_key=HF_TOKEN or "sk-placeholder", base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an AI airport immigration officer processing departing passengers.

=== AVAILABLE ACTIONS ===
Terminal decisions (advance to next passenger):
  clear     — All good, approve.
  hold      — Suspicious but not certain — secondary screening.
  deny      — Clear violation (expired doc, visa mismatch, etc).
  escalate  — Serious concern (watchlist, biometric fraud, unaccompanied minor, forged doc).

Information-gathering actions (do NOT advance queue, cost a little time):
  query_interpol    — Query the INTERPOL/national watchlist database for this passenger.
  verify_biometrics — Run biometric verification (face match score, fingerprint, document authenticity).
  request_document  — Ask passenger to present a specific document.

=== WHEN TO USE API ACTIONS ===
Call query_interpol when you see: "INTERPOL_QUERY_RECOMMENDED" flag, or the passenger name feels familiar/suspicious.
Call verify_biometrics when you see: "BIOMETRIC_SCAN_RECOMMENDED", "DOCUMENT_INSPECTION_RECOMMENDED" flags, or anything seems off.
You have a budget of 4 API calls per passenger. Use them wisely — they cost time.

=== DECISION RULES ===
1. Expired passport → deny (unless emergency travel doc present → clear)
2. Watchlist match score > 0.80 (from query_interpol) → escalate
3. Face match score < 0.70 OR fingerprint fail (from verify_biometrics) → escalate
4. Document not authentic (from verify_biometrics) → escalate
5. Visa type ≠ travel purpose → deny
6. Name mismatch between docs → hold
7. Overstay history → hold
8. Unaccompanied minor → escalate
9. Clean profile, no flags → clear

=== FAIRNESS NOTE ===
Your decisions are monitored for demographic bias. Do NOT factor nationality or gender into your
decision — base every decision strictly on documents, biometrics, and flags.

=== OUTPUT FORMAT ===
Respond ONLY with valid JSON, no markdown:
{
  "action_type": "clear|hold|deny|escalate|query_interpol|verify_biometrics|request_document",
  "passenger_id": "<id>",
  "reason": "<max 20 words>",
  "document_requested": null
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

SYSTEM FLAGS: {', '.join(obs.get('auto_flags', [])) or 'None'}
API calls used: {obs.get('api_calls_used', [])} | Remaining budget: {obs.get('api_calls_remaining', 4)}
Queue: {obs.get('queue_length', 0)} remaining | Time left: {obs.get('time_remaining', 0)}s
Last result: {obs.get('processing_result', '')}

What is your action?"""

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
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
    print(f"  Episode: {result['episode_id']} | Passengers: "
          f"{obs.get('queue_length', 0) + (1 if obs.get('current_passenger') else 0)}")

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

        print(f"\n  Step {step+1} | {p['name']} ({p['nationality']}) | Flags: {flags}")
        if api_used:
            print(f"    APIs called: {api_used}")

        action = call_llm(obs)
        if action is None:
            break

        print(f"    → {action['action_type'].upper()} | {action['reason']}")
        
        step_result = env_req("POST", "/step", {"action": action})
        obs = step_result["observation"]
        reward_dict = step_result["reward"]
        reward_val = reward_dict["total"]
        done = step_result["done"]
        cum_reward += reward_val
        all_rewards.append(reward_val)
        
        print(f"[STEP] step={step+1} action={json.dumps(action)} reward={reward_val:.2f} done={str(done).lower()} error=null", flush=True)
        print(f"    Reward: {reward_val:+.3f} | {reward_dict['explanation'][:70]}")
        step += 1
        time.sleep(0.25)

    grade = env_req("POST", "/grade", {})
    score_val = grade['score']
    success_val = str(score_val >= 0.5).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
    print(f"[END] success={success_val} steps={step} score={score_val:.3f} rewards={rewards_str}", flush=True)
    
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
