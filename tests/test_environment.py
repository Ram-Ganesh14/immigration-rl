"""
Test suite for Airport Immigration Processing Environment.
Run: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models.models import ImmigrationAction, ActionType
from server.environment import ImmigrationEnvironment, VALID_TASKS
from server.data_generator import PassengerGenerator
from graders.graders import run_grader, grade_task1, grade_task2


# ─── Environment tests ────────────────────────────────────────────────────────

class TestEnvironmentLifecycle:

    def setup_method(self):
        self.env = ImmigrationEnvironment()

    def test_reset_returns_observation(self):
        result = self.env.reset(task_id="task1_document_check", seed=42)
        assert result.observation is not None
        assert result.episode_id is not None
        assert result.task_id == "task1_document_check"

    def test_reset_produces_current_passenger(self):
        result = self.env.reset(task_id="task1_document_check", seed=42)
        assert result.observation.current_passenger is not None

    def test_reset_with_same_seed_is_deterministic(self):
        r1 = self.env.reset(task_id="task1_document_check", seed=42)
        r2 = self.env.reset(task_id="task1_document_check", seed=42)
        assert r1.observation.current_passenger.name == r2.observation.current_passenger.name

    def test_reset_with_different_seed_differs(self):
        r1 = self.env.reset(task_id="task1_document_check", seed=42)
        r2 = self.env.reset(task_id="task1_document_check", seed=99)
        # With different seeds, at least the episode IDs should differ
        assert r1.episode_id != r2.episode_id

    def test_all_task_ids_valid(self):
        for task_id in VALID_TASKS:
            result = self.env.reset(task_id=task_id, seed=1)
            assert result.task_id == task_id

    def test_invalid_task_raises_error(self):
        with pytest.raises(ValueError):
            self.env.reset(task_id="fake_task_xyz")

    def test_step_before_reset_raises(self):
        env = ImmigrationEnvironment()
        with pytest.raises(RuntimeError):
            action = ImmigrationAction(
                action_type=ActionType.CLEAR,
                passenger_id="xxx",
                reason="test"
            )
            env.step(action)

    def test_state_before_reset_raises(self):
        env = ImmigrationEnvironment()
        with pytest.raises(RuntimeError):
            env.state()

    def test_step_returns_step_result(self):
        result = self.env.reset(task_id="task1_document_check", seed=42)
        passenger = result.observation.current_passenger
        action = ImmigrationAction(
            action_type=ActionType.CLEAR,
            passenger_id=passenger.passenger_id,
            reason="All documents appear valid."
        )
        step_result = self.env.step(action)
        assert step_result.observation is not None
        assert isinstance(step_result.done, bool)
        assert step_result.reward is not None

    def test_reward_is_numeric(self):
        result = self.env.reset(task_id="task1_document_check", seed=42)
        passenger = result.observation.current_passenger
        action = ImmigrationAction(
            action_type=ActionType.CLEAR,
            passenger_id=passenger.passenger_id,
            reason="Test"
        )
        step_result = self.env.step(action)
        assert isinstance(step_result.reward.total, float)

    def test_state_returns_episode_state(self):
        self.env.reset(task_id="task1_document_check", seed=42)
        state = self.env.state()
        assert state.episode_id is not None
        assert state.step_count >= 0

    def test_done_when_queue_empty(self):
        result = self.env.reset(task_id="task1_document_check", seed=42)
        # Task 1 has 3 passengers — process them all
        for _ in range(10):
            obs = result.observation if _ == 0 else step_result.observation
            if not obs.current_passenger:
                break
            action = ImmigrationAction(
                action_type=ActionType.CLEAR,
                passenger_id=obs.current_passenger.passenger_id,
                reason="Test"
            )
            step_result = self.env.step(action)
            if step_result.done:
                break
        assert step_result.done is True


# ─── Data generator tests ─────────────────────────────────────────────────────

class TestDataGenerator:

    def test_clean_passenger_ground_truth(self):
        gen = PassengerGenerator(seed=1)
        p = gen.generate_clean_passenger()
        assert p.ground_truth_decision == "clear"
        assert len(p.documents) >= 2

    def test_expired_passport_ground_truth(self):
        gen = PassengerGenerator(seed=2)
        p = gen.generate_expired_passport()
        assert p.ground_truth_decision == "deny"
        assert "PASSPORT_EXPIRED" in p.flags

    def test_watchlist_ground_truth(self):
        gen = PassengerGenerator(seed=3)
        p = gen.generate_watchlist_hit()
        assert p.ground_truth_decision == "escalate"
        assert "WATCHLIST_MATCH" in p.flags
        assert p.watchlist_match.matched is True

    def test_emergency_doc_ground_truth(self):
        gen = PassengerGenerator(seed=4)
        p = gen.generate_emergency_travel_doc()
        assert p.ground_truth_decision == "clear"
        assert "PASSPORT_EXPIRED" in p.flags
        assert "EMERGENCY_TRAVEL_DOC_PRESENT" in p.flags

    def test_queue_size(self):
        gen = PassengerGenerator(seed=5)
        queue = gen.build_queue(n=5, difficulty="easy")
        assert len(queue) == 5

    def test_queue_deterministic(self):
        gen1 = PassengerGenerator(seed=7)
        gen2 = PassengerGenerator(seed=7)
        q1 = gen1.build_queue(n=4, difficulty="medium")
        q2 = gen2.build_queue(n=4, difficulty="medium")
        assert [p.name for p in q1] == [p.name for p in q2]

    def test_passenger_has_required_fields(self):
        gen = PassengerGenerator(seed=8)
        p = gen.generate_clean_passenger()
        assert p.passenger_id
        assert p.name
        assert p.nationality
        assert p.destination
        assert p.biometrics is not None
        assert p.watchlist_match is not None


# ─── Grader tests ─────────────────────────────────────────────────────────────

class TestGraders:

    def _perfect_log(self, n=3):
        return [
            {"passenger_id": str(i), "correct": True,
             "action": "clear", "ground_truth": "clear", "reward": 1.0}
            for i in range(n)
        ]

    def _all_wrong_log(self, n=3):
        return [
            {"passenger_id": str(i), "correct": False,
             "action": "clear", "ground_truth": "deny", "reward": -2.0}
            for i in range(n)
        ]

    def test_task1_perfect_score(self):
        result = grade_task1(self._perfect_log(), step_count=5, max_steps=30)
        assert result["score"] >= 0.9
        assert 0.0 <= result["score"] <= 1.0

    def test_task1_all_wrong(self):
        result = grade_task1(self._all_wrong_log(), step_count=5, max_steps=30)
        assert result["score"] < 0.5

    def test_task2_score_range(self):
        result = grade_task2(self._perfect_log(), step_count=10, max_steps=60)
        assert 0.0 <= result["score"] <= 1.0

    def test_grader_score_always_in_range(self):
        """Critical: grader must always return score in [0.0, 1.0]."""
        for log in [self._perfect_log(), self._all_wrong_log(), []]:
            for task_id in VALID_TASKS:
                state = {
                    "task_id": task_id,
                    "decision_log": log,
                    "step_count": 10,
                    "max_steps": 30,
                    "time_elapsed": 100,
                    "time_limit": 300,
                    "passengers_processed": len(log),
                    "passengers_total": max(len(log), 1),
                    "fairness_tracker": {},
                }
                result = run_grader(state)
                assert 0.0 <= result["score"] <= 1.0, (
                    f"Score out of range for task {task_id}: {result['score']}"
                )

    def test_grader_empty_log(self):
        result = grade_task1([], step_count=0, max_steps=30)
        assert result["score"] == 0.0

    def test_grader_has_explanation(self):
        result = grade_task1(self._perfect_log(), step_count=5, max_steps=30)
        assert "explanation" in result
        assert len(result["explanation"]) > 0


# ─── Integration test ─────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_episode_task1(self):
        """Run a complete task1 episode and verify grading works."""
        env = ImmigrationEnvironment()
        result = env.reset(task_id="task1_document_check", seed=42)

        done = False
        steps = 0
        while not done and steps < 50:
            obs = result.observation if steps == 0 else step_result.observation
            if not obs.current_passenger:
                break
            action = ImmigrationAction(
                action_type=ActionType.HOLD,  # always hold — mediocre agent
                passenger_id=obs.current_passenger.passenger_id,
                reason="Holding for review."
            )
            step_result = env.step(action)
            done = step_result.done
            steps += 1

        state = env.state()
        grade_result = run_grader(state.model_dump())

        assert 0.0 <= grade_result["score"] <= 1.0
        assert grade_result["task_id"] == "task1_document_check"

    def test_reproducible_scores(self):
        """Same seed, same model decisions → same grade."""
        def run_episode(seed):
            env = ImmigrationEnvironment()
            env.reset(task_id="task1_document_check", seed=seed)
            state = env.state()
            return run_grader(state.model_dump())["score"]

        s1 = run_episode(42)
        s2 = run_episode(42)
        assert s1 == s2
