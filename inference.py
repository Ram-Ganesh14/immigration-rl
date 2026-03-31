#!/usr/bin/env python3
"""
inference.py — Baseline inference script for Airport Immigration Processing Environment.

Uses the OpenAI client to run an LLM agent against all tasks.
Reads credentials from environment variables:
  API_BASE_URL  — LLM API endpoint
  MODEL_NAME    — Model identifier
  HF_TOKEN      — Hugging Face / API key

Usage:
  python inference.py
  python inference.py --task task1_document_check
  python inference.py --task all --seed 42

Runtime: < 20 minutes on vcpu=2, memory=8GB
"""

import os
import sys
import json
import time
import argparse
import requests
from openai import OpenAI

# ─── Config ──────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

TASKS = [
    "task1_document_check",
    "task2_flag_detection",
    "task3_queue_pressure",
]

MAX_STEPS_PER_TASK = 80
SEED = 42

# ─── OpenAI client ───────────────────────────────────────────────────────────

client = OpenAI(
    api_key=HF_TOKEN or "sk-placeholder",
    base_url=API_BASE_URL,
)

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI airport immigration officer. Your job is to process passengers efficiently and accurately.

For each passenger you receive, you must make ONE of these decisions:
- clear: Passenger is approved to proceed. All documents valid, no concerns.
- hold: Passenger needs secondary screening. Something is suspicious but not definitive.
- deny: Passenger must be denied boarding. Clear violation (expired docs, visa mismatch, etc.)
- escalate: Passenger must be escalated to a senior officer. Watchlist match, biometric failure, unaccompanied minor, document forgery suspected.
- request_document: Ask for additional document before deciding (use sparingly, max 2 times per passenger).

Rules:
1. EXPIRED passport → deny
2. WATCHLIST match (match_score > 0.80) → escalate
3. BIOMETRIC face match < 0.70 → escalate
4. VISA type doesn't match travel purpose → deny
5. Name mismatch between documents → hold
6. Overstay history → hold
7. Unaccompanied minor → escalate
8. Emergency travel doc + expired passport → clear (ETD overrides)
9. Forged document suspected → escalate

You MUST respond in this exact JSON format only:
{
  "action_type": "clear|hold|deny|escalate|request_document",
  "passenger_id": "<passenger_id from observation>",
  "reason": "<brief explanation max 20 words>",
  "document_requested": "<doc name if action_type is request_document, else null>"
}"""


# ─── LLM call ────────────────────────────────────────────────────────────────

def call_llm(observation: dict) -> dict:
    """Send observation to LLM, parse JSON action."""
    passenger = observation.get("current_passenger")
    if not passenger:
        return None

    # Build a clean prompt from the observation
    flags = observation.get("auto_flags", [])
    docs = []
    for d in passenger.get("documents", []):
        doc_info = (
            f"  - {d['doc_type']}: {d['doc_number']} | "
            f"Name: {d['name_on_doc']} | "
            f"Expiry: {d.get('expiry_date', 'N/A')} | "
            f"Visa type: {d.get('visa_type', 'N/A')}"
        )
        docs.append(doc_info)

    history = passenger.get("travel_history", [])
    history_str = ""
    for h in history:
        compliance = "✓" if h.get("visa_compliant") else "✗ NON-COMPLIANT"
        history_str += f"  - {h['country']} ({h['entry_date']} → {h.get('exit_date', 'N/A')}) {compliance}\n"

    bio = passenger.get("biometrics", {})
    wl = passenger.get("watchlist_match", {})
    special = passenger.get("special_circumstances", [])

    prompt = f"""CURRENT PASSENGER:
ID: {passenger['passenger_id']}
Name: {passenger['name']}
Nationality: {passenger['nationality']}
DOB: {passenger['date_of_birth']}
Destination: {passenger['destination']}
Travel purpose: {passenger['travel_purpose']}
Flight: {passenger['flight_number']}

DOCUMENTS:
{chr(10).join(docs)}

BIOMETRICS:
  Face match score: {bio.get('face_match_score', 'N/A')}
  Fingerprint match: {bio.get('fingerprint_match', 'N/A')}

WATCHLIST:
  Matched: {wl.get('matched', False)}
  Match score: {wl.get('match_score', 0.0)}
  Reason: {wl.get('match_reason', 'N/A')}

TRAVEL HISTORY:
{history_str if history_str else '  No prior travel history.'}

SPECIAL CIRCUMSTANCES: {', '.join(special) if special else 'None'}
AUTO-DETECTED FLAGS: {', '.join(flags) if flags else 'None'}

QUEUE: {observation.get('queue_length', 0)} passengers waiting.
TIME REMAINING: {observation.get('time_remaining', 0)} seconds.
LAST ACTION RESULT: {observation.get('processing_result', '')}

What is your decision?"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        print(f"    [WARN] LLM returned invalid JSON: {e}. Defaulting to hold.")
        return {
            "action_type": "hold",
            "passenger_id": passenger["passenger_id"],
            "reason": "Unable to parse response, defaulting to hold.",
            "document_requested": None
        }
    except Exception as e:
        print(f"    [ERROR] LLM call failed: {e}")
        return None


# ─── Environment interaction ──────────────────────────────────────────────────

def env_request(method: str, endpoint: str, payload: dict = None) -> dict:
    url = f"{ENV_URL}{endpoint}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=30)
        else:
            r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"[ERROR] Environment request failed: {e}")
        sys.exit(1)


def run_task(task_id: str, seed: int = SEED) -> dict:
    print(f"\n{'='*60}")
    print(f"  Task: {task_id}")
    print(f"  Seed: {seed} | Model: {MODEL_NAME}")
    print(f"{'='*60}")

    # Reset
    result = env_request("POST", "/reset", {"task_id": task_id, "seed": seed})
    obs = result["observation"]
    episode_id = result["episode_id"]
    print(f"  Episode: {episode_id}")
    print(f"  Passengers: {obs.get('queue_length', 0) + (1 if obs.get('current_passenger') else 0)}")

    step = 0
    done = False
    cumulative_reward = 0.0

    while not done and step < MAX_STEPS_PER_TASK:
        if not obs.get("current_passenger"):
            print("  No more passengers. Episode complete.")
            break

        passenger = obs["current_passenger"]
        print(f"\n  Step {step+1} | Passenger: {passenger['name']} ({passenger['nationality']})")
        print(f"    Flags: {obs.get('auto_flags', [])}")
        print(f"    Purpose: {passenger['travel_purpose']} → {passenger['destination']}")

        # Get LLM decision
        action_json = call_llm(obs)
        if action_json is None:
            print("    [ERROR] LLM failed. Skipping.")
            break

        print(f"    Decision: {action_json['action_type'].upper()} | {action_json['reason']}")

        # Send action to environment
        step_result = env_request("POST", "/step", {"action": action_json})
        obs = step_result["observation"]
        reward = step_result["reward"]
        done = step_result["done"]

        cumulative_reward += reward["total"]
        print(f"    Reward: {reward['total']:+.3f} | {reward['explanation']}")
        print(f"    Cumulative: {cumulative_reward:+.3f}")

        step += 1
        time.sleep(0.3)  # small delay to avoid rate limits

    # Grade the episode
    grade_result = env_request("POST", "/grade", {})

    print(f"\n  ── Final Grade ──────────────────────────────────")
    print(f"  Score:    {grade_result['score']:.4f} / 1.0000")
    print(f"  Details:  {grade_result.get('explanation', '')}")
    print(f"  Steps:    {step} | Cumulative reward: {cumulative_reward:+.3f}")

    return {
        "task_id": task_id,
        "score": grade_result["score"],
        "details": grade_result,
        "steps": step,
        "cumulative_reward": round(cumulative_reward, 3),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Airport Immigration Env — Baseline Inference")
    parser.add_argument("--task", default="all", help="Task ID or 'all'")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--env-url", default=ENV_URL)
    args = parser.parse_args()

    global ENV_URL
    ENV_URL = args.env_url

    # Health check
    health = env_request("GET", "/health")
    print(f"Environment: {health['environment']} v{health['version']} — {health['status']}")

    tasks_to_run = TASKS if args.task == "all" else [args.task]
    results = []

    start = time.time()
    for task_id in tasks_to_run:
        result = run_task(task_id, seed=args.seed)
        results.append(result)

    elapsed = time.time() - start

    # Summary
    print(f"\n{'='*60}")
    print("  BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  Seed:    {args.seed}")
    print(f"  Runtime: {elapsed:.1f}s\n")

    for r in results:
        bar = "█" * int(r["score"] * 20) + "░" * (20 - int(r["score"] * 20))
        print(f"  {r['task_id']:<30} [{bar}] {r['score']:.4f}")

    avg = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"\n  Average score: {avg:.4f}")
    print(f"{'='*60}\n")

    # Save results
    output = {
        "model": MODEL_NAME,
        "seed": args.seed,
        "runtime_seconds": round(elapsed, 1),
        "results": results,
        "average_score": round(avg, 4),
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("  Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
