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


# ─── Feature 2: RAG Policy Search ─────────────────────────────────────────────

class TestPolicySearch:
    def setup_method(self): self.env = ImmigrationEnvironment()

    def test_search_policy_returns_results(self):
        r = self.env.reset(task_id="task1_document_check", seed=42)
        p = r.observation.current_passenger
        sr = self.env.step(ImmigrationAction(
            action_type=ActionType.SEARCH_POLICY,
            passenger_id=p.passenger_id,
            reason="Check expired passport rules",
            policy_query="expired passport emergency travel"
        ))
        policies = sr.observation.queried_policies
        assert policies is not None
        assert len(policies) > 0
        assert "title" in policies[0]

    def test_search_policy_costs_api_budget(self):
        r = self.env.reset(task_id="task1_document_check", seed=42)
        p = r.observation.current_passenger
        sr = self.env.step(ImmigrationAction(
            action_type=ActionType.SEARCH_POLICY,
            passenger_id=p.passenger_id,
            policy_query="asylum"
        ))
        assert "search_policy" in sr.observation.api_calls_used
        assert sr.observation.api_calls_remaining == 3

    def test_duplicate_policy_search_penalised(self):
        r = self.env.reset(task_id="task1_document_check", seed=42)
        p = r.observation.current_passenger
        self.env.step(ImmigrationAction(
            action_type=ActionType.SEARCH_POLICY,
            passenger_id=p.passenger_id,
            policy_query="asylum"
        ))
        sr2 = self.env.step(ImmigrationAction(
            action_type=ActionType.SEARCH_POLICY,
            passenger_id=p.passenger_id,
            policy_query="transit"
        ))
        assert sr2.reward.total < 0, "Duplicate policy search should be penalised"

    def test_policy_search_finds_asylum_rules(self):
        r = self.env.reset(task_id="task1_document_check", seed=42)
        p = r.observation.current_passenger
        sr = self.env.step(ImmigrationAction(
            action_type=ActionType.SEARCH_POLICY,
            passenger_id=p.passenger_id,
            policy_query="asylum refugee seeker"
        ))
        policies = sr.observation.queried_policies
        assert any("asylum" in pol["title"].lower() for pol in policies)


# ─── Task 5: System Disruption ────────────────────────────────────────────────

class TestSystemDisruption:
    def setup_method(self): self.env = ImmigrationEnvironment()

    def test_task5_reset_ok(self):
        r = self.env.reset(task_id="task5_system_disruption", seed=42)
        assert r.task_id == "task5_system_disruption"
        assert r.observation.current_passenger is not None

    def test_task5_api_outage_at_step4(self):
        r = self.env.reset(task_id="task5_system_disruption", seed=42)
        # Process first 4 steps (terminal decisions to advance steps)
        for _ in range(4):
            obs = self.env._build_observation()
            if not obs.current_passenger:
                break
            self.env.step(ImmigrationAction(
                action_type=ActionType.CLEAR,
                passenger_id=obs.current_passenger.passenger_id
            ))
        # After step 4, outage should be active
        assert self.env._api_outage_active is True

    def test_task5_interpol_fails_during_outage(self):
        r = self.env.reset(task_id="task5_system_disruption", seed=42)
        # Get to step 4 (outage)
        for _ in range(4):
            obs = self.env._build_observation()
            if not obs.current_passenger:
                break
            self.env.step(ImmigrationAction(
                action_type=ActionType.CLEAR,
                passenger_id=obs.current_passenger.passenger_id
            ))
        # Try interpol during outage
        obs = self.env._build_observation()
        if obs.current_passenger:
            sr = self.env.step(ImmigrationAction(
                action_type=ActionType.QUERY_INTERPOL,
                passenger_id=obs.current_passenger.passenger_id
            ))
            assert sr.reward.total < 0, "Interpol should fail during outage"
            assert "OFFLINE" in sr.observation.processing_result or "outage" in sr.observation.processing_result.lower()

    def test_task5_grader_returns_valid_score(self):
        from graders.graders import grade_task5
        log = [
            {"correct": True, "action": "clear", "ground_truth": "clear",
             "api_calls_used": [], "api_outage_active": False, "policies_used": False}
            for _ in range(5)
        ]
        result = grade_task5(log, 10, 100, 120, 600, 5, 10)
        assert 0.0 < result["score"] < 1.0

    def test_task5_surge_passengers_added(self):
        r = self.env.reset(task_id="task5_system_disruption", seed=42)
        initial_total = self.env._state.passengers_total
        # Process 7 steps to trigger surge
        for _ in range(7):
            obs = self.env._build_observation()
            if not obs.current_passenger:
                break
            self.env.step(ImmigrationAction(
                action_type=ActionType.CLEAR,
                passenger_id=obs.current_passenger.passenger_id
            ))
        new_total = self.env._state.passengers_total
        assert new_total > initial_total, "Surge should add more passengers"


# ─── New Passenger Archetypes ─────────────────────────────────────────────────

class TestNewArchetypes:
    def setup_method(self):
        self.gen = PassengerGenerator(seed=42)

    def test_diplomatic_passport(self):
        p, i = self.gen.diplomatic_passport()
        assert i.ground_truth_decision == "clear"
        assert "DIPLOMATIC_PASSPORT" in p.flags
        assert "diplomatic_passport_holder" in p.special_circumstances

    def test_transit_passenger(self):
        p, i = self.gen.transit_passenger()
        assert i.ground_truth_decision == "clear"
        assert "TRANSIT_PASSENGER" in p.flags
        assert p.travel_purpose == "transit"

    def test_dual_nationality(self):
        p, i = self.gen.dual_nationality()
        assert i.ground_truth_decision == "hold"
        assert "DUAL_NATIONALITY" in p.flags
        assert len(p.documents) >= 3  # passport + visa + second passport

    def test_refugee_claimant(self):
        p, i = self.gen.refugee_claimant()
        assert i.ground_truth_decision == "escalate"
        assert "ASYLUM_CLAIM_DECLARED" in p.flags
        assert p.travel_purpose == "asylum"

    def test_new_types_in_hard_queue(self):
        profiles, internals = self.gen.build_queue(14, "hard")
        special = set()
        for p in profiles:
            for sc in p.special_circumstances:
                special.add(sc)
        assert len(special) > 0, "Hard queue should include new archetypes"

    def test_disruption_queue_builds(self):
        profiles, internals = self.gen.build_queue(10, "disruption")
        assert len(profiles) == 10
        assert len(internals) == 10

    def test_surge_passengers(self):
        surge_p, surge_i = self.gen.build_surge_passengers(3)
        assert len(surge_p) == 3
        assert len(surge_i) == 3


# ─── Intersectional Bias ──────────────────────────────────────────────────────

class TestIntersectionalBias:
    def test_gender_bias_detected(self):
        demo_log = [
            {"nationality": "German", "gender": "M", "ground_truth": "clear", "action": "clear"},
            {"nationality": "German", "gender": "F", "ground_truth": "clear", "action": "clear"},
            {"nationality": "German", "gender": "M", "ground_truth": "clear", "action": "clear"},
            # All female denied — gender bias
            {"nationality": "Nigerian", "gender": "F", "ground_truth": "clear", "action": "deny"},
            {"nationality": "Nigerian", "gender": "F", "ground_truth": "clear", "action": "deny"},
            {"nationality": "German", "gender": "F", "ground_truth": "clear", "action": "deny"},
        ]
        result = _demographic_bias_penalty(demo_log)
        # Should detect some form of bias
        assert result["penalty"] < 0.0

    def test_no_gender_bias_when_fair(self):
        demo_log = [
            {"nationality": "German", "gender": "M", "ground_truth": "clear", "action": "clear"},
            {"nationality": "German", "gender": "F", "ground_truth": "clear", "action": "clear"},
            {"nationality": "Nigerian", "gender": "M", "ground_truth": "clear", "action": "clear"},
            {"nationality": "Nigerian", "gender": "F", "ground_truth": "clear", "action": "clear"},
        ]
        result = _demographic_bias_penalty(demo_log)
        assert result["penalty"] == 0.0

    def test_adversarial_queue_gender_balanced(self):
        gen = PassengerGenerator(seed=42)
        profiles, internals = gen.build_queue(10, "adversarial")
        genders = set(i.gender for i in internals)
        # Should have both M and F
        assert len(genders) >= 2, "Adversarial queue must have balanced gender"


# ─── Explainability ───────────────────────────────────────────────────────────

class TestExplainability:
    def setup_method(self): self.env = ImmigrationEnvironment()

    def test_explain_returns_none_before_decision(self):
        self.env.reset(task_id="task1_document_check", seed=42)
        assert self.env.get_last_decision_info() is None

    def test_explain_returns_info_after_decision(self):
        r = self.env.reset(task_id="task1_document_check", seed=42)
        p = r.observation.current_passenger
        self.env.step(ImmigrationAction(action_type=ActionType.CLEAR, passenger_id=p.passenger_id))
        info = self.env.get_last_decision_info()
        assert info is not None
        assert "feature_importance" in info
        assert "key_factors" in info
        assert "confidence" in info
        assert "ground_truth" in info

    def test_explain_feature_importance_sums_to_one(self):
        r = self.env.reset(task_id="task1_document_check", seed=42)
        p = r.observation.current_passenger
        self.env.step(ImmigrationAction(action_type=ActionType.CLEAR, passenger_id=p.passenger_id))
        info = self.env.get_last_decision_info()
        total = sum(info["feature_importance"].values())
        assert 0.95 <= total <= 1.05, f"Feature importance should sum to ~1.0, got {total}"

    def test_explain_correct_decision_has_high_confidence(self):
        r = self.env.reset(task_id="task1_document_check", seed=42)
        p = r.observation.current_passenger
        # Get ground truth for this passenger
        gt = self.env._current_internal.ground_truth_decision
        self.env.step(ImmigrationAction(
            action_type=ActionType(gt), passenger_id=p.passenger_id, reason="test"
        ))
        info = self.env.get_last_decision_info()
        assert info["confidence"] > 0.8


# ─── RL Mechanic 1: Adversarial Queue Escalation ─────────────────────────────

class TestAdversarialEscalation:
    def setup_method(self): self.env = ImmigrationEnvironment()

    def test_sloppy_clears_tracked(self):
        """Clearing without API calls increments the sloppy counter."""
        r = self.env.reset(task_id="task3_queue_pressure", seed=42)
        p = r.observation.current_passenger
        self.env.step(ImmigrationAction(
            action_type=ActionType.CLEAR, passenger_id=p.passenger_id, reason="fast"
        ))
        assert self.env._consecutive_sloppy_clears >= 1

    def test_api_call_resets_sloppy_counter(self):
        """Using an API before deciding resets the sloppy counter."""
        r = self.env.reset(task_id="task3_queue_pressure", seed=42)
        p = r.observation.current_passenger
        # Sloppy clear
        self.env.step(ImmigrationAction(
            action_type=ActionType.CLEAR, passenger_id=p.passenger_id, reason="fast"
        ))
        assert self.env._consecutive_sloppy_clears >= 1
        # Now use an API on the next passenger
        p2 = self.env._current_passenger
        if p2:
            self.env.step(ImmigrationAction(
                action_type=ActionType.QUERY_INTERPOL, passenger_id=p2.passenger_id, reason="check"
            ))
            gt = self.env._current_internal.ground_truth_decision
            self.env.step(ImmigrationAction(
                action_type=ActionType(gt), passenger_id=p2.passenger_id, reason="after api"
            ))
            assert self.env._consecutive_sloppy_clears == 0

    def test_three_sloppy_clears_triggers_escalation(self):
        """After 3 consecutive sloppy clears, adversarial surge is injected."""
        r = self.env.reset(task_id="task3_queue_pressure", seed=42)
        original_total = self.env._state.passengers_total
        # Do 3 sloppy clears in a row
        for _ in range(3):
            p = self.env._current_passenger
            if p is None:
                break
            self.env.step(ImmigrationAction(
                action_type=ActionType.CLEAR, passenger_id=p.passenger_id, reason="sloppy"
            ))
        assert self.env._adversarial_escalation_active is True
        assert self.env._state.passengers_total == original_total + 2

    def test_escalation_only_fires_once(self):
        """Adversarial escalation should not re-fire after it's already active."""
        r = self.env.reset(task_id="task3_queue_pressure", seed=42)
        # Trigger escalation
        for _ in range(3):
            p = self.env._current_passenger
            if p is None:
                break
            self.env.step(ImmigrationAction(
                action_type=ActionType.CLEAR, passenger_id=p.passenger_id, reason="sloppy"
            ))
        total_after_first = self.env._state.passengers_total
        # Do 3 more sloppy clears
        for _ in range(3):
            p = self.env._current_passenger
            if p is None:
                break
            self.env.step(ImmigrationAction(
                action_type=ActionType.CLEAR, passenger_id=p.passenger_id, reason="sloppy again"
            ))
        # Should not have added more passengers
        assert self.env._state.passengers_total == total_after_first

    def test_state_exposes_escalation_flag(self):
        """EpisodeState should report adversarial_escalation_active via state()."""
        r = self.env.reset(task_id="task3_queue_pressure", seed=42)
        s = self.env.state()
        assert s.adversarial_escalation_active is False
        # Trigger it
        for _ in range(3):
            p = self.env._current_passenger
            if p is None:
                break
            self.env.step(ImmigrationAction(
                action_type=ActionType.CLEAR, passenger_id=p.passenger_id, reason="sloppy"
            ))
        s = self.env.state()
        assert s.adversarial_escalation_active is True


# ─── RL Mechanic 2: API Reliability Degradation ──────────────────────────────

class TestAPIDegradation:
    def setup_method(self): self.env = ImmigrationEnvironment()

    def test_no_degradation_with_low_api_usage(self):
        """API should not degrade if agent queries sparingly."""
        r = self.env.reset(task_id="task3_queue_pressure", seed=42)
        # Process 4 passengers, query API on only 1
        for i in range(4):
            p = self.env._current_passenger
            if p is None:
                break
            if i == 0:
                self.env.step(ImmigrationAction(
                    action_type=ActionType.QUERY_INTERPOL, passenger_id=p.passenger_id
                ))
            gt = self.env._current_internal.ground_truth_decision
            self.env.step(ImmigrationAction(
                action_type=ActionType(gt), passenger_id=p.passenger_id, reason="test"
            ))
        assert self.env._api_degraded is False

    def test_degradation_triggers_with_heavy_api_usage(self):
        """API should degrade when usage rate >= 1.5 after 4+ passengers."""
        r = self.env.reset(task_id="task3_queue_pressure", seed=42)
        # Process passengers, querying both APIs on each one
        for i in range(5):
            p = self.env._current_passenger
            if p is None:
                break
            self.env.step(ImmigrationAction(
                action_type=ActionType.QUERY_INTERPOL, passenger_id=p.passenger_id
            ))
            self.env.step(ImmigrationAction(
                action_type=ActionType.VERIFY_BIOMETRICS, passenger_id=p.passenger_id
            ))
            gt = self.env._current_internal.ground_truth_decision
            self.env.step(ImmigrationAction(
                action_type=ActionType(gt), passenger_id=p.passenger_id, reason="test"
            ))
        assert self.env._api_degraded is True

    def test_state_exposes_degradation_flag(self):
        """EpisodeState should report api_degraded via state()."""
        r = self.env.reset(task_id="task3_queue_pressure", seed=42)
        s = self.env.state()
        assert s.api_degraded is False

    def test_degraded_biometrics_shows_degraded_label(self):
        """When degraded, the biometric system label should include DEGRADED."""
        r = self.env.reset(task_id="task3_queue_pressure", seed=42)
        # Force degradation by querying heavily
        for i in range(5):
            p = self.env._current_passenger
            if p is None:
                break
            self.env.step(ImmigrationAction(
                action_type=ActionType.QUERY_INTERPOL, passenger_id=p.passenger_id
            ))
            self.env.step(ImmigrationAction(
                action_type=ActionType.VERIFY_BIOMETRICS, passenger_id=p.passenger_id
            ))
            gt = self.env._current_internal.ground_truth_decision
            self.env.step(ImmigrationAction(
                action_type=ActionType(gt), passenger_id=p.passenger_id, reason="test"
            ))
        # Now query biometrics on next passenger — should be degraded
        p = self.env._current_passenger
        if p:
            self.env.step(ImmigrationAction(
                action_type=ActionType.VERIFY_BIOMETRICS, passenger_id=p.passenger_id
            ))
            bio = p.queried_biometrics
            assert bio is not None
            assert "DEGRADED" in bio.get("system", "")
