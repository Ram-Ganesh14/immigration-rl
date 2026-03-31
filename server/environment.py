"""
ImmigrationEnvironment — core OpenEnv environment class.
Implements reset() / step() / state() as per OpenEnv spec.
"""

import uuid
import hashlib
import time
from typing import Optional, Dict, Any, List

from models.models import (
    ImmigrationObservation, ImmigrationAction, ImmigrationReward,
    EpisodeState, StepResult, ResetResult,
    PassengerProfile, ActionType, RiskLevel
)
from server.data_generator import PassengerGenerator


# ─── Task configs ─────────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "task1_document_check": {
        "queue_size": 3,
        "difficulty": "easy",
        "time_limit": 300,
        "max_steps": 30,
        "description": "Basic document verification — small queue, clear issues only.",
    },
    "task2_flag_detection": {
        "queue_size": 6,
        "difficulty": "medium",
        "time_limit": 600,
        "max_steps": 60,
        "description": "Red flag detection — mixed queue, hidden anomalies.",
    },
    "task3_queue_pressure": {
        "queue_size": 15,
        "difficulty": "hard",
        "time_limit": 900,
        "max_steps": 150,
        "description": "High-pressure queue — edge cases, time pressure, adversarial profiles.",
    },
    "task4_adversarial": {
        "queue_size": 8,
        "difficulty": "hard",
        "time_limit": 600,
        "max_steps": 100,
        "description": "Expert mode — passengers provide contradictory follow-up information.",
    },
}

VALID_TASKS = list(TASK_CONFIGS.keys())

# ─── Reward constants ─────────────────────────────────────────────────────────

REWARD_CORRECT_CLEAR     = +1.0
REWARD_CORRECT_HOLD      = +1.2
REWARD_CORRECT_DENY      = +1.2
REWARD_CORRECT_ESCALATE  = +1.5
REWARD_WRONG_DENY        = -2.0   # false positive — blocking innocent person
REWARD_WRONG_CLEAR       = -3.0   # false negative — letting threat through
REWARD_WRONG_DECISION    = -1.5   # other wrong decisions
REWARD_OVER_ESCALATE     = -0.3   # escalating when unnecessary
REWARD_LOOP_PENALTY      = -0.1   # per extra action beyond reasonable limit
REWARD_SPEED_MAX         = +0.3   # bonus for fast correct decisions
FAIRNESS_PENALTY         = -0.5   # inconsistent decisions for same profile type


class ImmigrationEnvironment:
    def __init__(self):
        self._state: Optional[EpisodeState] = None
        self._queue: List[PassengerProfile] = []
        self._processed: List[PassengerProfile] = []
        self._current_passenger: Optional[PassengerProfile] = None
        self._passenger_action_count: int = 0
        self._start_time: float = 0.0
        self._task_config: Dict[str, Any] = {}

    # ─── Public API ───────────────────────────────────────────────────────────

    def reset(self, task_id: str = "task1_document_check", seed: Optional[int] = None) -> ResetResult:
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {VALID_TASKS}")

        seed = seed if seed is not None else int(time.time()) % 100000
        self._task_config = TASK_CONFIGS[task_id]

        gen = PassengerGenerator(seed=seed)
        self._queue = gen.build_queue(
            n=self._task_config["queue_size"],
            difficulty=self._task_config["difficulty"]
        )
        self._processed = []
        self._passenger_action_count = 0
        self._start_time = time.time()

        episode_id = str(uuid.uuid4())[:12]
        self._state = EpisodeState(
            episode_id=episode_id,
            task_id=task_id,
            seed=seed,
            step_count=0,
            max_steps=self._task_config["max_steps"],
            time_elapsed=0,
            time_limit=self._task_config["time_limit"],
            passengers_processed=0,
            passengers_total=len(self._queue),
            cumulative_reward=0.0,
            done=False,
        )

        self._current_passenger = self._queue.pop(0) if self._queue else None
        obs = self._build_observation()

        return ResetResult(
            observation=obs,
            episode_id=episode_id,
            task_id=task_id
        )

    def step(self, action: ImmigrationAction) -> StepResult:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._state.step_count += 1
        self._state.time_elapsed = int(time.time() - self._start_time)
        self._passenger_action_count += 1

        reward, feedback = self._process_action(action)
        self._state.cumulative_reward += reward.total

        self._state.action_history.append({
            "step": self._state.step_count,
            "passenger_id": action.passenger_id,
            "action_type": action.action_type,
            "reason": action.reason,
            "reward": reward.total,
        })

        # Check done conditions
        done = (
            self._state.step_count >= self._state.max_steps
            or (self._current_passenger is None and len(self._queue) == 0)
            or self._state.time_elapsed >= self._state.time_limit
        )
        self._state.done = done

        obs = self._build_observation(processing_result=feedback)
        obs.step_count = self._state.step_count

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "episode_id": self._state.episode_id,
                "cumulative_reward": round(self._state.cumulative_reward, 3),
                "passengers_processed": self._state.passengers_processed,
                "passengers_remaining": len(self._queue) + (1 if self._current_passenger else 0),
                "time_elapsed": self._state.time_elapsed,
            }
        )

    def state(self) -> EpisodeState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        self._state.time_elapsed = int(time.time() - self._start_time)
        self._state.current_passenger_id = (
            self._current_passenger.passenger_id if self._current_passenger else None
        )
        return self._state

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _process_action(self, action: ImmigrationAction) -> tuple[ImmigrationReward, str]:
        passenger = self._current_passenger

        # Validate passenger_id matches
        if passenger and action.passenger_id != passenger.passenger_id:
            reward = ImmigrationReward(
                total=-0.5,
                breakdown={"invalid_passenger_id": -0.5},
                explanation="Action references wrong passenger ID."
            )
            return reward, "⚠ Invalid passenger ID in action."

        # Handle request_document — agent asks for more info (doesn't advance queue)
        if action.action_type == ActionType.REQUEST_DOCUMENT:
            if self._passenger_action_count > 3:
                penalty = REWARD_LOOP_PENALTY
                reward = ImmigrationReward(
                    total=penalty,
                    loop_penalty=penalty,
                    breakdown={"excessive_requests": penalty},
                    explanation="Too many document requests for the same passenger."
                )
                return reward, f"⚠ Excessive document requests. Penalised."
            reward = ImmigrationReward(
                total=0.0,
                breakdown={"document_request": 0.0},
                explanation=f"Requested: {action.document_requested}. Passenger presents it."
            )
            return reward, f"Passenger presents {action.document_requested}. Review and decide."

        # Terminal action — evaluate decision
        if not passenger:
            reward = ImmigrationReward(
                total=-0.2,
                breakdown={"no_passenger": -0.2},
                explanation="No passenger to act on."
            )
            return reward, "No current passenger."

        reward = self._evaluate_decision(action, passenger)

        # Log decision for fairness tracking
        profile_hash = self._profile_hash(passenger)
        if profile_hash not in self._state.fairness_tracker:
            self._state.fairness_tracker[profile_hash] = []
        self._state.fairness_tracker[profile_hash].append(action.action_type)

        # Check fairness (same profile, different decision)
        fairness_penalty = 0.0
        decisions = self._state.fairness_tracker[profile_hash]
        if len(decisions) > 1 and len(set(decisions)) > 1:
            fairness_penalty = FAIRNESS_PENALTY
            reward.fairness_penalty = fairness_penalty
            reward.total += fairness_penalty
            reward.breakdown["fairness_penalty"] = fairness_penalty
            reward.explanation += " | Inconsistent decision for similar passenger profile."

        # Log processed passenger
        self._state.decision_log.append({
            "passenger_id": passenger.passenger_id,
            "name": passenger.name,
            "action": action.action_type,
            "correct": action.action_type == passenger.ground_truth_decision,
            "ground_truth": passenger.ground_truth_decision,
            "reward": round(reward.total, 3),
        })

        # Advance queue
        self._state.passengers_processed += 1
        self._processed.append(passenger)
        self._current_passenger = self._queue.pop(0) if self._queue else None
        self._passenger_action_count = 0

        return reward, self._format_feedback(action, passenger, reward)

    def _evaluate_decision(self, action: ImmigrationAction, passenger: PassengerProfile) -> ImmigrationReward:
        gt = passenger.ground_truth_decision
        act = action.action_type

        # Speed bonus: faster decisions on obvious cases get bonus
        time_per_passenger = self._state.time_elapsed / max(1, self._state.passengers_processed + 1)
        time_limit = self._task_config["time_limit"]
        speed_ratio = 1.0 - min(1.0, time_per_passenger / (time_limit / self._state.passengers_total))
        speed_bonus = round(speed_ratio * REWARD_SPEED_MAX, 3) if act == gt else 0.0

        decision_reward = 0.0
        explanation = ""

        if act == gt:
            if act == ActionType.CLEAR:
                decision_reward = REWARD_CORRECT_CLEAR
            elif act == ActionType.HOLD:
                decision_reward = REWARD_CORRECT_HOLD
            elif act == ActionType.DENY:
                decision_reward = REWARD_CORRECT_DENY
            elif act == ActionType.ESCALATE:
                decision_reward = REWARD_CORRECT_ESCALATE
            explanation = f"✓ Correct decision: {act}."
        else:
            # Wrong decision — severity depends on what was missed
            if act == ActionType.CLEAR and gt in ["deny", "escalate"]:
                decision_reward = REWARD_WRONG_CLEAR  # worst — let threat through
                explanation = f"✗ Cleared a passenger who should be {gt}. Critical error."
            elif act == ActionType.DENY and gt == "clear":
                decision_reward = REWARD_WRONG_DENY   # false positive — innocent denied
                explanation = f"✗ Denied an innocent passenger. False positive."
            elif act == ActionType.ESCALATE and gt in ["clear", "hold"]:
                decision_reward = REWARD_OVER_ESCALATE
                explanation = f"✗ Over-escalated. Should have been {gt}."
            else:
                decision_reward = REWARD_WRONG_DECISION
                explanation = f"✗ Wrong decision. Should be {gt}, got {act}."

        total = round(decision_reward + speed_bonus, 3)

        return ImmigrationReward(
            total=total,
            decision_reward=decision_reward,
            speed_bonus=speed_bonus,
            breakdown={
                "decision": decision_reward,
                "speed_bonus": speed_bonus,
            },
            explanation=explanation
        )

    def _build_observation(self, processing_result: str = "") -> ImmigrationObservation:
        queue_summary = []
        for p in self._queue[:5]:  # show next 5 in queue
            queue_summary.append({
                "passenger_id": p.passenger_id,
                "name": p.name,
                "nationality": p.nationality,
                "destination": p.destination,
                "flags": p.flags,
            })

        time_remaining = max(0, self._task_config.get("time_limit", 300) - int(
            time.time() - self._start_time
        ))

        return ImmigrationObservation(
            current_passenger=self._current_passenger,
            queue_length=len(self._queue),
            queue_summary=queue_summary,
            time_remaining=time_remaining,
            step_count=self._state.step_count if self._state else 0,
            max_steps=self._task_config.get("max_steps", 30),
            processing_result=processing_result,
            auto_flags=self._current_passenger.flags if self._current_passenger else [],
            episode_id=self._state.episode_id if self._state else "",
            task_id=self._state.task_id if self._state else "",
            documents_requested=[],
            secondary_screening_available=True,
        )

    def _profile_hash(self, passenger: PassengerProfile) -> str:
        """Hash key fields to detect similar profiles for fairness checking."""
        key = f"{passenger.nationality}:{passenger.travel_purpose}:{passenger.risk_level}"
        return hashlib.md5(key.encode()).hexdigest()[:8]

    def _format_feedback(self, action: ImmigrationAction, passenger: PassengerProfile, reward: ImmigrationReward) -> str:
        icon = "✓" if action.action_type == passenger.ground_truth_decision else "✗"
        remaining = len(self._queue) + (1 if self._current_passenger else 0)
        return (
            f"{icon} {passenger.name} → {action.action_type.upper()} | "
            f"Reward: {reward.total:+.2f} | "
            f"Remaining: {remaining} passengers"
        )

    @property
    def passengers_total(self) -> int:
        return self._state.passengers_total if self._state else 0
