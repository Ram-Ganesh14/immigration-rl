"""Full test suite covering Features 1 and 3."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models.models import ImmigrationAction, ActionType
from server.environment import ImmigrationEnvironment, VALID_TASKS
from server.data_generator import PassengerGenerator
from graders.graders import run_grader, grade_task1, grade_task2, _demographic_bias_penalty


# ─── Environment lifecycle ────────────────────────────────────────────────────

class TestLifecycle:
    def setup_method(self): self.env = ImmigrationEnvironment()

    def test_reset_returns_observation(self):
        r = self.env.reset(task_id="task1_document_check", seed=42)
        assert r.observation is not None and r.episode_id

    def test_reset_has_passenger(self):
        r = self.env.reset(task_id="task1_document_check", seed=42)
        assert r.observation.current_passenger is not None

    def test_reset_deterministic(self):
        r1 = self.env.reset(task_id="task1_document_check", seed=42)
        r2 = self.env.reset(task_id="task1_document_check", seed=42)
        assert r1.observation.current_passenger.name == r2.observation.current_passenger.name

    def test_all_tasks_valid(self):
        for tid in VALID_TASKS:
            r = self.env.reset(task_id=tid, seed=1)
            assert r.task_id == tid

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            self.env.reset(task_id="nonexistent_task")

    def test_step_before_reset_raises(self):
        env = ImmigrationEnvironment()
        with pytest.raises(RuntimeError):
            env.step(ImmigrationAction(action_type=ActionType.CLEAR, passenger_id="x"))

    def test_state_before_reset_raises(self):
        env = ImmigrationEnvironment()
        with pytest.raises(RuntimeError):
            env.state()

    def test_step_returns_result(self):
        r = self.env.reset(task_id="task1_document_check", seed=42)
        p = r.observation.current_passenger
        sr = self.env.step(ImmigrationAction(action_type=ActionType.CLEAR, passenger_id=p.passenger_id))
        assert sr.observation is not None
        assert isinstance(sr.done, bool)
        assert isinstance(sr.reward.total, float)

    def test_done_when_queue_empty(self):
        r = self.env.reset(task_id="task1_document_check", seed=42)
        done = False
        step_result = None
        for _ in range(20):
            obs = r.observation if step_result is None else step_result.observation
            if not obs.current_passenger:
                break
            step_result = self.env.step(ImmigrationAction(
                action_type=ActionType.CLEAR, passenger_id=obs.current_passenger.passenger_id
            ))
            done = step_result.done
            if done: break
        assert done is True


# ─── Feature 1: Hidden API tests ──────────────────────────────────────────────

class TestHiddenAPIs:
    def setup_method(self): self.env = ImmigrationEnvironment()

    def test_passenger_has_no_biometrics_by_default(self):
        r = self.env.reset(task_id="task1_document_check", seed=42)
        p = r.observation.current_passenger
        assert p.queried_biometrics is None, "Biometrics should be hidden until queried"

    def test_passenger_has_no_watchlist_by_default(self):
        r = self.env.reset(task_id="task1_document_check", seed=42)
        p = r.observation.current_passenger
        assert p.queried_watchlist is None, "Watchlist should be hidden until queried"

    def test_verify_biometrics_reveals_data(self):
        r = self.env.reset(task_id="task1_document_check", seed=42)
        p = r.observation.current_passenger
        sr = self.env.step(ImmigrationAction(
            action_type=ActionType.VERIFY_BIOMETRICS, passenger_id=p.passenger_id
        ))
        bio = sr.observation.current_passenger.queried_biometrics
        assert bio is not None
        assert "face_match_score" in bio
        assert "fingerprint_match" in bio
        assert "document_authentic" in bio

    def test_query_interpol_reveals_watchlist(self):
        r = self.env.reset(task_id="task1_document_check", seed=42)
        p = r.observation.current_passenger
        sr = self.env.step(ImmigrationAction(
            action_type=ActionType.QUERY_INTERPOL, passenger_id=p.passenger_id
        ))
        wl = sr.observation.current_passenger.queried_watchlist
        assert wl is not None
        assert "matched" in wl
        assert "match_score" in wl

    def test_api_calls_tracked_in_observation(self):
        r = self.env.reset(task_id="task1_document_check", seed=42)
        p = r.observation.current_passenger
        assert r.observation.api_calls_remaining == 4
        sr = self.env.step(ImmigrationAction(
            action_type=ActionType.VERIFY_BIOMETRICS, passenger_id=p.passenger_id
        ))
        assert "verify_biometrics" in sr.observation.api_calls_used
        assert sr.observation.api_calls_remaining == 3

    def test_duplicate_api_call_penalised(self):
        r = self.env.reset(task_id="task1_document_check", seed=42)
        p = r.observation.current_passenger
        self.env.step(ImmigrationAction(action_type=ActionType.VERIFY_BIOMETRICS, passenger_id=p.passenger_id))
        sr2 = self.env.step(ImmigrationAction(action_type=ActionType.VERIFY_BIOMETRICS, passenger_id=p.passenger_id))
        assert sr2.reward.total < 0, "Duplicate API call should be penalised"

    def test_api_then_decision_works(self):
        r = self.env.reset(task_id="task2_flag_detection", seed=99)
        p = r.observation.current_passenger
        self.env.step(ImmigrationAction(action_type=ActionType.QUERY_INTERPOL, passenger_id=p.passenger_id))
        sr = self.env.step(ImmigrationAction(action_type=ActionType.CLEAR, passenger_id=p.passenger_id))
        # After terminal action, queue advances
        assert sr.observation.current_passenger != p or sr.done

    def test_api_calls_reset_per_passenger(self):
        r = self.env.reset(task_id="task2_flag_detection", seed=42)
        p = r.observation.current_passenger
        self.env.step(ImmigrationAction(action_type=ActionType.VERIFY_BIOMETRICS, passenger_id=p.passenger_id))
        # Make terminal decision to advance to next passenger
        sr = self.env.step(ImmigrationAction(action_type=ActionType.CLEAR, passenger_id=p.passenger_id))
        # New passenger should have fresh API budget
        assert sr.observation.api_calls_used == []
        assert sr.observation.api_calls_remaining == 4


# ─── Feature 3: Demographic bias tests ────────────────────────────────────────

class TestDemographicBias:

    def test_no_bias_when_consistent(self):
        demo_log = [
            {"nationality": "German",  "gender": "M", "ground_truth": "clear", "action": "clear", "correct": True},
            {"nationality": "German",  "gender": "F", "ground_truth": "clear", "action": "clear", "correct": True},
            {"nationality": "Nigerian","gender": "M", "ground_truth": "clear", "action": "clear", "correct": True},
            {"nationality": "Nigerian","gender": "F", "ground_truth": "clear", "action": "clear", "correct": True},
        ]
        result = _demographic_bias_penalty(demo_log)
        assert result["penalty"] == 0.0

    def test_bias_detected_when_one_group_over_denied(self):
        demo_log = [
            {"nationality": "German",  "gender": "M", "ground_truth": "clear", "action": "clear", "correct": True},
            {"nationality": "German",  "gender": "F", "ground_truth": "clear", "action": "clear", "correct": True},
            {"nationality": "German",  "gender": "M", "ground_truth": "clear", "action": "clear", "correct": True},
            # Nigerian passengers (clean) all wrongly denied
            {"nationality": "Nigerian","gender": "M", "ground_truth": "clear", "action": "deny", "correct": False},
            {"nationality": "Nigerian","gender": "F", "ground_truth": "clear", "action": "deny", "correct": False},
            {"nationality": "Nigerian","gender": "M", "ground_truth": "clear", "action": "deny", "correct": False},
        ]
        result = _demographic_bias_penalty(demo_log)
        assert result["penalty"] < 0.0, "Should detect bias and apply penalty"
        assert len(result["bias_found"]) > 0

    def test_task4_demo_log_populated(self):
        env = ImmigrationEnvironment()
        env.reset(task_id="task4_adversarial", seed=42)
        state = env.state()
        # Process all passengers
        for _ in range(20):
            s = env.state()
            if s.done or not s.current_passenger_id:
                break
            # find current passenger by id
            obs = env._build_observation()
            if not obs.current_passenger:
                break
            env.step(ImmigrationAction(action_type=ActionType.CLEAR,
                                       passenger_id=obs.current_passenger.passenger_id))
        state = env.state()
        assert len(state.demographic_log) > 0

    def test_grader_score_in_range_all_tasks(self):
        perfect_log = [
            {"correct": True, "action": "clear", "ground_truth": "clear",
             "passenger_id": str(i), "reward": 1.0, "api_calls_used": []}
            for i in range(4)
        ]
        for tid in VALID_TASKS:
            result = run_grader({
                "task_id": tid, "decision_log": perfect_log,
                "step_count": 5, "max_steps": 30, "time_elapsed": 60,
                "time_limit": 300, "passengers_processed": 4, "passengers_total": 4,
                "fairness_tracker": {}, "demographic_log": [],
            })
            assert 0.0 <= result["score"] <= 1.0, f"Score out of range for {tid}: {result['score']}"


# ─── Data generator tests ─────────────────────────────────────────────────────

class TestDataGenerator:

    def test_returns_paired_tuples(self):
        gen = PassengerGenerator(seed=1)
        profiles, internals = gen.build_queue(n=3, difficulty="easy")
        assert len(profiles) == 3 and len(internals) == 3

    def test_passenger_ids_match(self):
        gen = PassengerGenerator(seed=2)
        profiles, internals = gen.build_queue(n=5, difficulty="medium")
        for p, i in zip(profiles, internals):
            assert p.passenger_id == i.passenger_id

    def test_internal_not_exposed_in_profile(self):
        gen = PassengerGenerator(seed=3)
        p, i = gen.clean()
        assert not hasattr(p, "is_authentic"), "is_authentic must not be on PassengerProfile"
        assert not hasattr(p, "face_match_score"), "face_match_score must not be on PassengerProfile"
        assert not hasattr(p, "watchlist_matched"), "watchlist_matched must not be on PassengerProfile"

    def test_biometrics_hidden_by_default(self):
        gen = PassengerGenerator(seed=4)
        p, _ = gen.clean()
        assert p.queried_biometrics is None
        assert p.queried_watchlist is None

    def test_adversarial_queue_has_demographics(self):
        gen = PassengerGenerator(seed=5)
        profiles, internals = gen.build_queue(n=10, difficulty="adversarial")
        nats = {i.nationality for i in internals}
        assert len(nats) >= 2, "Adversarial queue should have multiple nationalities"

    def test_deterministic(self):
        g1, g2 = PassengerGenerator(seed=7), PassengerGenerator(seed=7)
        p1, _ = g1.build_queue(4, "medium")
        p2, _ = g2.build_queue(4, "medium")
        assert [p.name for p in p1] == [p.name for p in p2]

    def test_watchlist_hit_ground_truth(self):
        gen = PassengerGenerator(seed=8)
        p, i = gen.watchlist_hit()
        assert i.ground_truth_decision == "escalate"
        assert i.watchlist_matched is True
        assert p.queried_watchlist is None  # must be hidden

    def test_forged_doc_ground_truth(self):
        gen = PassengerGenerator(seed=9)
        p, i = gen.forged_document()
        assert i.ground_truth_decision == "escalate"
        assert i.is_authentic is False
        assert p.queried_biometrics is None  # hidden


# ─── Integration ──────────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_episode_with_api_calls(self):
        env = ImmigrationEnvironment()
        r = env.reset(task_id="task2_flag_detection", seed=42)
        done, sr = False, None
        steps = 0
        while not done and steps < 100:
            obs = r.observation if sr is None else sr.observation
            if not obs.current_passenger:
                break
            pid = obs.current_passenger.passenger_id
            flags = obs.auto_flags
            # Call APIs if flags suggest it
            if "BIOMETRIC_SCAN_RECOMMENDED" in flags and "verify_biometrics" not in obs.api_calls_used:
                sr = env.step(ImmigrationAction(action_type=ActionType.VERIFY_BIOMETRICS, passenger_id=pid))
                steps += 1
                continue
            if "INTERPOL_QUERY_RECOMMENDED" in flags and "query_interpol" not in obs.api_calls_used:
                sr = env.step(ImmigrationAction(action_type=ActionType.QUERY_INTERPOL, passenger_id=pid))
                steps += 1
                continue
            sr = env.step(ImmigrationAction(action_type=ActionType.CLEAR, passenger_id=pid))
            done = sr.done
            steps += 1

        state = env.state()
        grade = run_grader(state.model_dump())
        assert 0.0 <= grade["score"] <= 1.0

    def test_reproducible_grades(self):
        def run(seed):
            env = ImmigrationEnvironment()
            env.reset(task_id="task1_document_check", seed=seed)
            return run_grader(env.state().model_dump())["score"]
        assert run(42) == run(42)

    def test_task4_bias_analysis_in_grade(self):
        env = ImmigrationEnvironment()
        env.reset(task_id="task4_adversarial", seed=42)
        state = env.state()
        grade = run_grader(state.model_dump())
        assert "bias_analysis" in grade
        assert "demographic_bias_penalty" in grade
