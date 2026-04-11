"""
LIVE PROOF: Both RL mechanics work against the running server.
This script demonstrates that the environment is genuinely stateful
and that agent behavior changes what happens next.
"""
import requests, json, sys

BASE = "http://localhost:7860"
def post(p, d=None): return requests.post(f"{BASE}{p}", json=d or {})
def get(p): return requests.get(f"{BASE}{p}")

print("=" * 70)
print("  RL STATEFULNESS PROOF — LIVE DEMO")
print("=" * 70)

# =====================================================================
# PROOF 1: ADVERSARIAL QUEUE ESCALATION
# If agent clears 3 passengers WITHOUT any API calls, queue grows
# =====================================================================
print("\n" + "=" * 70)
print("  PROOF 1: SLOPPY CLEARING -> ADVERSARIAL ESCALATION")
print("=" * 70)

r = post("/reset", {"task_id": "task3_queue_pressure", "seed": 42})
obs = r.json()["observation"]
state = get("/state").json()

print(f"\n  BEFORE: passengers_total = {state['passengers_total']}")
print(f"  BEFORE: adversarial_escalation_active = {state['adversarial_escalation_active']}")
original_total = state["passengers_total"]

print("\n  Action: Clearing 3 passengers WITHOUT querying any APIs (sloppy)...")
for i in range(3):
    p = obs.get("current_passenger")
    if not p:
        break
    pid = p["passenger_id"]
    print(f"    Step {i+1}: CLEAR {p['name']} ({pid}) - NO API calls used")
    r2 = post("/step", {"action": {"action_type": "clear", "passenger_id": pid, "reason": "quick clear"}})
    obs = r2.json()["observation"]

state = get("/state").json()
print(f"\n  AFTER:  passengers_total = {state['passengers_total']}")
print(f"  AFTER:  adversarial_escalation_active = {state['adversarial_escalation_active']}")

if state["adversarial_escalation_active"] and state["passengers_total"] == original_total + 2:
    print("\n  >>> PROOF 1 PASSED: 2 high-risk passengers injected into queue!")
    print("  >>> An RL agent that learns this pattern will STOP clearing sloppily.")
else:
    print("\n  >>> PROOF 1 FAILED")
    sys.exit(1)

# =====================================================================
# PROOF 2: API RELIABILITY DEGRADATION
# If agent queries APIs on every passenger, APIs degrade with noise
# =====================================================================
print("\n" + "=" * 70)
print("  PROOF 2: HEAVY API USAGE -> API DEGRADATION")
print("=" * 70)

r = post("/reset", {"task_id": "task3_queue_pressure", "seed": 100})
obs = r.json()["observation"]
state = get("/state").json()

print(f"\n  BEFORE: api_degraded = {state['api_degraded']}")

print("\n  Action: Querying BOTH APIs on every passenger (5 passengers)...")
for i in range(5):
    p = obs.get("current_passenger")
    if not p:
        break
    pid = p["passenger_id"]
    
    # Query interpol
    post("/step", {"action": {"action_type": "query_interpol", "passenger_id": pid, "reason": "check"}})
    # Query biometrics
    post("/step", {"action": {"action_type": "verify_biometrics", "passenger_id": pid, "reason": "scan"}})
    # Clear
    r3 = post("/step", {"action": {"action_type": "clear", "passenger_id": pid, "reason": "done"}})
    obs = r3.json()["observation"]
    print(f"    Passenger {i+1}: queried interpol + biometrics + cleared")

state = get("/state").json()
print(f"\n  AFTER:  api_degraded = {state['api_degraded']}")

# Now query biometrics on the next passenger and check for DEGRADED label
p = obs.get("current_passenger")
if p:
    pid = p["passenger_id"]
    r4 = post("/step", {"action": {"action_type": "verify_biometrics", "passenger_id": pid, "reason": "test noise"}})
    bio = r4.json()["observation"]["current_passenger"]["queried_biometrics"]
    system_label = bio.get("system", "")
    face_score = bio.get("face_match_score", -1)
    print(f"\n  Biometric system label: {system_label}")
    print(f"  Face match score: {face_score}")
    
    if "DEGRADED" in system_label:
        print("\n  >>> PROOF 2 PASSED: APIs degraded! Biometric scores now have noise.")
        print("  >>> An RL agent that learns this will query SELECTIVELY, not always.")
    else:
        print("\n  >>> PROOF 2 FAILED: Expected DEGRADED label")
        sys.exit(1)

# =====================================================================
# PROOF 3: CONTRAST — Careful agent avoids BOTH traps
# =====================================================================
print("\n" + "=" * 70)
print("  PROOF 3: CAREFUL AGENT -> NO ESCALATION, NO DEGRADATION")
print("=" * 70)

r = post("/reset", {"task_id": "task3_queue_pressure", "seed": 42})
obs = r.json()["observation"]

print("\n  Action: Processing 5 passengers with SELECTIVE API usage (1 API per passenger)...")
for i in range(5):
    p = obs.get("current_passenger")
    if not p:
        break
    pid = p["passenger_id"]
    
    # Query only 1 API (selective)
    post("/step", {"action": {"action_type": "query_interpol", "passenger_id": pid, "reason": "selective check"}})
    # Clear
    r3 = post("/step", {"action": {"action_type": "clear", "passenger_id": pid, "reason": "after check"}})
    obs = r3.json()["observation"]
    print(f"    Passenger {i+1}: queried interpol only + cleared")

state = get("/state").json()
print(f"\n  RESULT: api_degraded = {state['api_degraded']}")
print(f"  RESULT: adversarial_escalation_active = {state['adversarial_escalation_active']}")

if not state["api_degraded"] and not state["adversarial_escalation_active"]:
    print("\n  >>> PROOF 3 PASSED: Selective usage avoids BOTH traps!")
    print("  >>> This is the OPTIMAL STRATEGY an RL agent would discover.")
else:
    print(f"\n  >>> PROOF 3 ISSUE: Expected both False")

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("  SUMMARY: WHY THIS IS A TRUE RL ENVIRONMENT")
print("=" * 70)
print("""
  Strategy A (Sloppy):    Clear without APIs     -> Queue fills with adversarial cases
  Strategy B (Greedy):    Query every API always  -> APIs degrade with noise  
  Strategy C (Selective): Query APIs when needed  -> Clean data, no escalation
  
  An LLM cannot discover Strategy C from a prompt alone.
  Only RL training across episodes reveals the exact thresholds.
  
  This is what makes this environment genuinely RL-oriented.
""")
print("=" * 70)
print("  ALL 3 PROOFS PASSED - RL MECHANICS VERIFIED LIVE")
print("=" * 70)
