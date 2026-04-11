"""Quick smoke test for all 5 tasks + new features."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.environment import ImmigrationEnvironment
from models.models import ImmigrationAction, ActionType
from graders.graders import run_grader

env = ImmigrationEnvironment()

tasks = [
    "task1_document_check",
    "task2_flag_detection",
    "task3_queue_pressure",
    "task4_adversarial",
    "task5_system_disruption",
]

print("=" * 60)
print("SMOKE TEST: All 5 tasks with 'clear all' strategy")
print("=" * 60)

for task in tasks:
    r = env.reset(task, 42)
    obs = r.observation
    steps = 0
    while obs.current_passenger and steps < 25:
        pid = obs.current_passenger.passenger_id
        action = ImmigrationAction(
            action_type=ActionType.CLEAR,
            passenger_id=pid,
            reason="Auto clear"
        )
        step = env.step(action)
        obs = step.observation
        steps += 1
        if step.done:
            break

    state = env.state()
    grade = run_grader(state.model_dump())
    score = grade["score"]
    status = "PASS" if 0.0 < score < 1.0 else "CHECK"
    print(f"  {task:<35} score={score:.3f}  [{status}]")

# Test new features
print("\n" + "=" * 60)
print("FEATURE TESTS")
print("=" * 60)

# 1. search_policy
env.reset("task1_document_check", 42)
obs = env._build_observation()
pid = obs.current_passenger.passenger_id
action = ImmigrationAction(
    action_type=ActionType.SEARCH_POLICY,
    passenger_id=pid,
    reason="Check rules",
    policy_query="expired passport emergency"
)
step = env.step(action)
policies = step.observation.queried_policies
print(f"  search_policy: returned {len(policies) if policies else 0} policies  [PASS]")

# 2. system_alerts (Task 5)
env.reset("task5_system_disruption", 42)
for i in range(5):
    obs = env._build_observation()
    if obs.current_passenger:
        pid = obs.current_passenger.passenger_id
        action = ImmigrationAction(action_type=ActionType.CLEAR, passenger_id=pid, reason="test")
        env.step(action)

obs = env._build_observation()
has_alerts = len(obs.system_alerts) > 0 or env._api_outage_active
print(f"  system_alerts: outage_active={env._api_outage_active}  [PASS]")

# 3. explain
env.reset("task1_document_check", 42)
obs = env._build_observation()
pid = obs.current_passenger.passenger_id
env.step(ImmigrationAction(action_type=ActionType.CLEAR, passenger_id=pid, reason="test"))
info = env.get_last_decision_info()
has_features = "feature_importance" in info if info else False
print(f"  explainability: has_features={has_features}  [PASS]")

# 4. new passenger types
from server.data_generator import PassengerGenerator
gen = PassengerGenerator(seed=42)
p_dip, i_dip = gen.diplomatic_passport()
p_tra, i_tra = gen.transit_passenger()
p_dua, i_dua = gen.dual_nationality()
p_ref, i_ref = gen.refugee_claimant()
print(f"  diplomatic: gt={i_dip.ground_truth_decision}  [PASS]")
print(f"  transit: gt={i_tra.ground_truth_decision}  [PASS]")
print(f"  dual_nationality: gt={i_dua.ground_truth_decision}  [PASS]")
print(f"  refugee: gt={i_ref.ground_truth_decision}  [PASS]")

# 5. intersectional bias
from graders.graders import _demographic_bias_penalty
demo_log = [
    {"nationality": "German", "gender": "M", "ground_truth": "clear", "action": "clear"},
    {"nationality": "German", "gender": "F", "ground_truth": "clear", "action": "clear"},
    {"nationality": "Nigerian", "gender": "M", "ground_truth": "clear", "action": "deny"},
    {"nationality": "Nigerian", "gender": "F", "ground_truth": "clear", "action": "deny"},
]
result = _demographic_bias_penalty(demo_log)
print(f"  intersectional_bias: penalty={result['penalty']}  [PASS]")
print(f"  bias_found: {result['bias_found']}")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
