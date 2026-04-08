"""
ImmigrationEnvironment — OpenEnv environment with:
  - Feature 1: Hidden biometrics/watchlist — agent must call query_interpol / verify_biometrics
  - Feature 3: Demographic bias tracking and penalty
"""

import uuid
import hashlib
import time
from typing import Optional, Dict, Any, List, Tuple

from models.models import (
    ImmigrationObservation, ImmigrationAction, ImmigrationReward,
    EpisodeState, StepResult, ResetResult,
    PassengerProfile, _PassengerInternalData, ActionType, RiskLevel
)
from server.data_generator import PassengerGenerator


TASK_CONFIGS = {
    "task1_document_check": {
        "queue_size": 3, "difficulty": "easy",
        "time_limit": 300, "max_steps": 30,
        "description": "Basic document verification — small queue, clear issues only.",
    },
    "task2_flag_detection": {
        "queue_size": 6, "difficulty": "medium",
        "time_limit": 600, "max_steps": 60,
        "description": "Red flag detection — mixed queue, hidden anomalies.",
    },
    "task3_queue_pressure": {
        "queue_size": 15, "difficulty": "hard",
        "time_limit": 900, "max_steps": 150,
        "description": "High-pressure queue — edge cases, time pressure, adversarial profiles.",
    },
    "task4_adversarial": {
        "queue_size": 10, "difficulty": "adversarial",
        "time_limit": 700, "max_steps": 120,
        "description": "Adversarial dataset — biased nationality mix, tests fairness and consistency.",
    },
}

VALID_TASKS = list(TASK_CONFIGS.keys())

# ─── Reward constants ─────────────────────────────────────────────────────────
REWARD_CORRECT_CLEAR    = +1.0
REWARD_CORRECT_HOLD     = +1.2
REWARD_CORRECT_DENY     = +1.2
REWARD_CORRECT_ESCALATE = +1.5
REWARD_WRONG_CLEAR      = -3.0   # missed threat — worst
REWARD_WRONG_DENY       = -2.0   # innocent denied
REWARD_OVER_ESCALATE    = -0.3
REWARD_WRONG_DECISION   = -1.5
REWARD_SPEED_MAX        = +0.3
REWARD_LOOP_PENALTY     = -0.1
FAIRNESS_PENALTY        = -0.5
API_CALL_COST           = -0.05  # small time penalty per API call (speed vs accuracy tradeoff)
API_CALL_MAX            = 4      # per passenger


class ImmigrationEnvironment:
    def __init__(self):
        self._state: Optional[EpisodeState] = None
        self._queue: List[PassengerProfile] = []
        self._internal_queue: List[_PassengerInternalData] = []
        self._processed: List[PassengerProfile] = []
        self._current_passenger: Optional[PassengerProfile] = None
        self._current_internal: Optional[_PassengerInternalData] = None
        self._passenger_action_count: int = 0
        self._api_calls_used: List[str] = []
        self._start_time: float = 0.0
        self._task_config: Dict[str, Any] = {}

    # ─── Public API ───────────────────────────────────────────────────────────

    def reset(self, task_id: str = "task1_document_check", seed: Optional[int] = None) -> ResetResult:
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {VALID_TASKS}")

        seed = seed if seed is not None else int(time.time()) % 100000
        self._task_config = TASK_CONFIGS[task_id]

        gen = PassengerGenerator(seed=seed)
        passengers, internals = gen.build_queue(
            n=self._task_config["queue_size"],
            difficulty=self._task_config["difficulty"]
        )

        self._queue = passengers
        self._internal_queue = internals
        self._processed = []
        self._passenger_action_count = 0
        self._api_calls_used = []
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

        self._advance_queue()
        obs = self._build_observation()
        return ResetResult(observation=obs, episode_id=episode_id, task_id=task_id)

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

    # ─── Action processing ────────────────────────────────────────────────────

    def _process_action(self, action: ImmigrationAction) -> Tuple[ImmigrationReward, str]:
        passenger = self._current_passenger
        internal = self._current_internal

        if passenger and action.passenger_id != passenger.passenger_id:
            return ImmigrationReward(
                total=-0.5,
                breakdown={"invalid_passenger_id": -0.5},
                explanation="Action references wrong passenger ID."
            ), "⚠ Invalid passenger ID."

        # ── Feature 1: query_interpol ──────────────────────────────────────
        if action.action_type == ActionType.QUERY_INTERPOL:
            return self._handle_query_interpol(passenger, internal)

        # ── Feature 1: verify_biometrics ───────────────────────────────────
        if action.action_type == ActionType.VERIFY_BIOMETRICS:
            return self._handle_verify_biometrics(passenger, internal)

        # ── request_document ───────────────────────────────────────────────
        if action.action_type == ActionType.REQUEST_DOCUMENT:
            if self._passenger_action_count > 3:
                return ImmigrationReward(
                    total=REWARD_LOOP_PENALTY,
                    loop_penalty=REWARD_LOOP_PENALTY,
                    breakdown={"excessive_requests": REWARD_LOOP_PENALTY},
                    explanation="Too many requests for same passenger."
                ), "⚠ Excessive document requests. Penalised."
            return ImmigrationReward(
                total=0.0,
                breakdown={"document_request": 0.0},
                explanation=f"Document requested: {action.document_requested}."
            ), f"Passenger presents {action.document_requested}. Review and decide."

        # ── Terminal decisions ─────────────────────────────────────────────
        if not passenger or not internal:
            return ImmigrationReward(
                total=-0.2, breakdown={"no_passenger": -0.2},
                explanation="No current passenger."
            ), "No current passenger."

        reward = self._evaluate_decision(action, passenger, internal)

        # Fairness tracking (Feature 3)
        profile_hash = self._profile_hash(internal)
        if profile_hash not in self._state.fairness_tracker:
            self._state.fairness_tracker[profile_hash] = []
        self._state.fairness_tracker[profile_hash].append(action.action_type)

        decisions = self._state.fairness_tracker[profile_hash]
        if len(decisions) > 1 and len(set(decisions)) > 1:
            reward.fairness_penalty = FAIRNESS_PENALTY
            reward.total += FAIRNESS_PENALTY
            reward.breakdown["fairness_penalty"] = FAIRNESS_PENALTY
            reward.explanation += " | Inconsistent decision for similar profile."

        # Demographic log (Feature 3)
        self._state.demographic_log.append({
            "passenger_id": passenger.passenger_id,
            "nationality": internal.nationality,
            "gender": internal.gender,
            "ground_truth": internal.ground_truth_decision,
            "action": action.action_type,
            "correct": action.action_type == internal.ground_truth_decision,
            "risk_level": internal.risk_level,
        })

        # Decision log
        self._state.decision_log.append({
            "passenger_id": passenger.passenger_id,
            "name": passenger.name,
            "action": action.action_type,
            "correct": action.action_type == internal.ground_truth_decision,
            "ground_truth": internal.ground_truth_decision,
            "reward": round(reward.total, 3),
            "api_calls_used": list(self._api_calls_used),
        })

        # Advance to next passenger
        self._state.passengers_processed += 1
        self._processed.append(passenger)
        self._advance_queue()

        return reward, self._format_feedback(action, passenger, internal, reward)

    def _handle_query_interpol(
        self, passenger: Optional[PassengerProfile], internal: Optional[_PassengerInternalData]
    ) -> Tuple[ImmigrationReward, str]:
        """Reveal watchlist data. Charges a small time cost."""
        if not passenger or not internal:
            return ImmigrationReward(total=0.0, explanation="No passenger."), "No passenger."

        api_reward = API_CALL_COST
        calls_used = len(self._api_calls_used)

        if calls_used >= API_CALL_MAX:
            return ImmigrationReward(
                total=REWARD_LOOP_PENALTY,
                loop_penalty=REWARD_LOOP_PENALTY,
                explanation="API call budget exhausted for this passenger."
            ), "⚠ API budget exhausted. Make your final decision."

        if "query_interpol" in self._api_calls_used:
            return ImmigrationReward(
                total=REWARD_LOOP_PENALTY,
                loop_penalty=REWARD_LOOP_PENALTY,
                explanation="Interpol already queried for this passenger."
            ), "⚠ Already queried Interpol for this passenger."

        self._api_calls_used.append("query_interpol")

        # Inject results into the passenger's queried_watchlist field
        passenger.queried_watchlist = {
            "matched": internal.watchlist_matched,
            "match_score": internal.watchlist_score,
            "match_reason": internal.watchlist_reason,
            "database": "INTERPOL + NATIONAL_WATCHLIST",
            "queried_at_step": self._state.step_count,
        }

        result_str = (
            f"INTERPOL QUERY RESULT for {passenger.name}: "
            f"Match={internal.watchlist_matched}, "
            f"Score={internal.watchlist_score:.2f}, "
            f"Reason={internal.watchlist_reason or 'None'}"
        )

        return ImmigrationReward(
            total=api_reward,
            api_cost=api_reward,
            breakdown={"api_call_interpol": api_reward},
            explanation=f"Interpol queried. Cost: {api_reward}."
        ), result_str

    def _handle_verify_biometrics(
        self, passenger: Optional[PassengerProfile], internal: Optional[_PassengerInternalData]
    ) -> Tuple[ImmigrationReward, str]:
        """Reveal biometric data. Charges a small time cost."""
        if not passenger or not internal:
            return ImmigrationReward(total=0.0, explanation="No passenger."), "No passenger."

        calls_used = len(self._api_calls_used)
        if calls_used >= API_CALL_MAX:
            return ImmigrationReward(
                total=REWARD_LOOP_PENALTY,
                loop_penalty=REWARD_LOOP_PENALTY,
                explanation="API call budget exhausted."
            ), "⚠ API budget exhausted."

        if "verify_biometrics" in self._api_calls_used:
            return ImmigrationReward(
                total=REWARD_LOOP_PENALTY,
                loop_penalty=REWARD_LOOP_PENALTY,
                explanation="Biometrics already verified for this passenger."
            ), "⚠ Already verified biometrics for this passenger."

        self._api_calls_used.append("verify_biometrics")

        passenger.queried_biometrics = {
            "face_match_score": internal.face_match_score,
            "fingerprint_match": internal.fingerprint_match,
            "document_authentic": internal.is_authentic,
            "system": "BIOMETRIC_VERIFICATION_SYSTEM_v2",
            "queried_at_step": self._state.step_count,
        }

        result_str = (
            f"BIOMETRIC RESULT for {passenger.name}: "
            f"Face match={internal.face_match_score:.2f}, "
            f"Fingerprint={'OK' if internal.fingerprint_match else 'FAIL'}, "
            f"Document authentic={'YES' if internal.is_authentic else 'NO — FORGERY DETECTED'}"
        )

        return ImmigrationReward(
            total=API_CALL_COST,
            api_cost=API_CALL_COST,
            breakdown={"api_call_biometrics": API_CALL_COST},
            explanation=f"Biometrics verified. Cost: {API_CALL_COST}."
        ), result_str

    def _evaluate_decision(
        self, action: ImmigrationAction,
        passenger: PassengerProfile,
        internal: _PassengerInternalData
    ) -> ImmigrationReward:
        gt = internal.ground_truth_decision
        act = action.action_type

        # Speed bonus
        time_per_passenger = self._state.time_elapsed / max(1, self._state.passengers_processed + 1)
        time_limit = self._task_config["time_limit"]
        speed_ratio = 1.0 - min(1.0, time_per_passenger / (time_limit / max(1, self._state.passengers_total)))
        speed_bonus = round(speed_ratio * REWARD_SPEED_MAX, 3) if act == gt else 0.0

        decision_reward = 0.0
        explanation = ""

        if act == gt:
            rewards_map = {
                ActionType.CLEAR: REWARD_CORRECT_CLEAR,
                ActionType.HOLD: REWARD_CORRECT_HOLD,
                ActionType.DENY: REWARD_CORRECT_DENY,
                ActionType.ESCALATE: REWARD_CORRECT_ESCALATE,
            }
            decision_reward = rewards_map.get(act, REWARD_CORRECT_CLEAR)
            explanation = f"✓ Correct: {act}."
        else:
            if act == ActionType.CLEAR and gt in ["deny", "escalate"]:
                decision_reward = REWARD_WRONG_CLEAR
                explanation = f"✗ Cleared a {gt} passenger. Critical miss."
            elif act == ActionType.DENY and gt == "clear":
                decision_reward = REWARD_WRONG_DENY
                explanation = f"✗ Denied an innocent passenger. False positive."
            elif act == ActionType.ESCALATE and gt in ["clear", "hold"]:
                decision_reward = REWARD_OVER_ESCALATE
                explanation = f"✗ Over-escalated. Should be {gt}."
            else:
                decision_reward = REWARD_WRONG_DECISION
                explanation = f"✗ Wrong. Expected {gt}, got {act}."

        # Bonus for correct use of APIs before hard decisions
        api_bonus = 0.0
        if gt in ["escalate", "deny"] and act == gt:
            if "query_interpol" in self._api_calls_used or "verify_biometrics" in self._api_calls_used:
                api_bonus = 0.1  # rewarded for doing due diligence
                explanation += " | +0.1 due-diligence bonus for querying APIs."

        total = round(decision_reward + speed_bonus + api_bonus, 3)

        return ImmigrationReward(
            total=total,
            decision_reward=decision_reward,
            speed_bonus=speed_bonus,
            breakdown={
                "decision": decision_reward,
                "speed_bonus": speed_bonus,
                "api_diligence_bonus": api_bonus,
            },
            explanation=explanation
        )

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _advance_queue(self):
        if self._queue:
            self._current_passenger = self._queue.pop(0)
            self._current_internal = self._internal_queue.pop(0)
        else:
            self._current_passenger = None
            self._current_internal = None
        self._passenger_action_count = 0
        self._api_calls_used = []

    def _build_observation(self, processing_result: str = "") -> ImmigrationObservation:
        queue_summary = []
        for p in self._queue[:5]:
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
            api_calls_used=list(self._api_calls_used),
            api_calls_remaining=max(0, API_CALL_MAX - len(self._api_calls_used)),
        )

    def _profile_hash(self, internal: _PassengerInternalData) -> str:
        key = f"{internal.nationality}:{internal.risk_level}"
        return hashlib.md5(key.encode()).hexdigest()[:8]

    def _format_feedback(
        self, action: ImmigrationAction,
        passenger: PassengerProfile,
        internal: _PassengerInternalData,
        reward: ImmigrationReward
    ) -> str:
        icon = "✓" if action.action_type == internal.ground_truth_decision else "✗"
        remaining = len(self._queue) + (1 if self._current_passenger else 0)
        return (
            f"{icon} {passenger.name} → {action.action_type.upper()} | "
            f"Reward: {reward.total:+.2f} | Remaining: {remaining}"
        )

    @property
    def passengers_total(self) -> int:
        return self._state.passengers_total if self._state else 0
