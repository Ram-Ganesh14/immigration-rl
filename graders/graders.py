"""
Agent graders for all tasks.
Each grader takes the episode decision_log and returns a score 0.0–1.0.
Deterministic and reproducible.
"""

from typing import List, Dict, Any


def _base_accuracy(decision_log: List[Dict[str, Any]]) -> float:
    """Simple decision accuracy across all processed passengers."""
    if not decision_log:
        return 0.0
    correct = sum(1 for d in decision_log if d.get("correct", False))
    return correct / len(decision_log)


def _weighted_accuracy(decision_log: List[Dict[str, Any]]) -> float:
    """
    Weighted accuracy — wrong clear of dangerous passenger penalised more
    than wrong deny of innocent one. Reflects real-world priorities.
    """
    if not decision_log:
        return 0.0

    total_weight = 0
    earned = 0.0

    for d in decision_log:
        gt = d.get("ground_truth", "")
        act = d.get("action", "")
        correct = d.get("correct", False)

        # Weight by ground truth severity
        if gt in ["escalate", "deny"]:
            weight = 2.0
        elif gt == "hold":
            weight = 1.5
        else:
            weight = 1.0

        total_weight += weight
        if correct:
            earned += weight
        elif act == "clear" and gt in ["deny", "escalate"]:
            earned -= weight * 0.5  # extra penalty for missing dangerous cases

    raw = earned / total_weight if total_weight > 0 else 0.0
    return max(0.0, min(1.0, raw))


# ─── Task 1 Grader ────────────────────────────────────────────────────────────

def grade_task1(decision_log: List[Dict[str, Any]], step_count: int, max_steps: int) -> Dict[str, Any]:
    """
    Task 1: Basic document verification.
    Score = accuracy (simple — small queue, obvious issues only).
    """
    accuracy = _base_accuracy(decision_log)

    # Efficiency bonus: under 50% of max steps used
    efficiency = 1.0 if step_count <= max_steps * 0.5 else 0.8

    score = round(accuracy * efficiency, 3)
    score = max(0.0, min(1.0, score))

    return {
        "task_id": "task1_document_check",
        "score": score,
        "accuracy": accuracy,
        "efficiency": efficiency,
        "passengers_judged": len(decision_log),
        "correct_decisions": sum(1 for d in decision_log if d.get("correct")),
        "explanation": (
            f"Accuracy: {accuracy:.1%}, Efficiency: {efficiency:.1%}. "
            f"Final score: {score:.3f}"
        )
    }


# ─── Task 2 Grader ────────────────────────────────────────────────────────────

def grade_task2(decision_log: List[Dict[str, Any]], step_count: int, max_steps: int) -> Dict[str, Any]:
    """
    Task 2: Red flag detection.
    Score = weighted accuracy (prioritises catching flags) + flag recall.
    """
    weighted_acc = _weighted_accuracy(decision_log)

    # Flag recall: did agent catch all flagged passengers?
    flagged_gt = [d for d in decision_log if d.get("ground_truth") in ["hold", "deny", "escalate"]]
    flagged_correct = [d for d in flagged_gt if d.get("correct")]
    flag_recall = len(flagged_correct) / len(flagged_gt) if flagged_gt else 1.0

    # False positive rate: innocent passengers wrongly denied
    innocent_gt = [d for d in decision_log if d.get("ground_truth") == "clear"]
    false_positives = [d for d in innocent_gt if d.get("action") == "deny"]
    fp_penalty = len(false_positives) / len(innocent_gt) if innocent_gt else 0.0

    score = round(0.5 * weighted_acc + 0.4 * flag_recall - 0.1 * fp_penalty, 3)
    score = max(0.0, min(1.0, score))

    return {
        "task_id": "task2_flag_detection",
        "score": score,
        "weighted_accuracy": weighted_acc,
        "flag_recall": flag_recall,
        "false_positive_rate": fp_penalty,
        "flagged_caught": len(flagged_correct),
        "flagged_total": len(flagged_gt),
        "explanation": (
            f"Weighted accuracy: {weighted_acc:.1%}, "
            f"Flag recall: {flag_recall:.1%}, "
            f"False positive rate: {fp_penalty:.1%}. "
            f"Final score: {score:.3f}"
        )
    }


# ─── Task 3 Grader ────────────────────────────────────────────────────────────

def grade_task3(
    decision_log: List[Dict[str, Any]],
    step_count: int,
    max_steps: int,
    time_elapsed: int,
    time_limit: int,
    passengers_processed: int,
    passengers_total: int
) -> Dict[str, Any]:
    """
    Task 3: High-pressure queue management.
    Score = weighted accuracy + throughput + time efficiency.
    Penalises both false negatives (missed threats) and excessive stalling.
    """
    weighted_acc = _weighted_accuracy(decision_log)

    # Throughput: proportion of passengers processed
    throughput = passengers_processed / passengers_total if passengers_total > 0 else 0.0

    # Time efficiency: did they finish before time limit?
    time_efficiency = 1.0 - min(1.0, time_elapsed / time_limit) if time_limit > 0 else 0.5

    # Edge case bonus: emergency travel doc handled correctly?
    edge_cases = [
        d for d in decision_log
        if d.get("ground_truth") in ["clear", "escalate"]
        and d.get("correct")
    ]
    edge_case_bonus = min(0.1, len(edge_cases) * 0.02)

    score = round(
        0.45 * weighted_acc
        + 0.30 * throughput
        + 0.20 * time_efficiency
        + edge_case_bonus,
        3
    )
    score = max(0.0, min(1.0, score))

    return {
        "task_id": "task3_queue_pressure",
        "score": score,
        "weighted_accuracy": weighted_acc,
        "throughput": throughput,
        "time_efficiency": time_efficiency,
        "edge_case_bonus": edge_case_bonus,
        "passengers_processed": passengers_processed,
        "passengers_total": passengers_total,
        "time_elapsed": time_elapsed,
        "explanation": (
            f"Weighted accuracy: {weighted_acc:.1%}, "
            f"Throughput: {throughput:.1%}, "
            f"Time efficiency: {time_efficiency:.1%}. "
            f"Final score: {score:.3f}"
        )
    }


# ─── Task 4 Grader ────────────────────────────────────────────────────────────

def grade_task4(
    decision_log: List[Dict[str, Any]],
    step_count: int,
    max_steps: int,
    fairness_tracker: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Task 4: Adversarial + fairness evaluation.
    Score = weighted accuracy + consistency penalty + over-escalation check.
    """
    weighted_acc = _weighted_accuracy(decision_log)

    # Consistency: same profile type should get same decision
    inconsistency_count = 0
    for profile_hash, decisions in fairness_tracker.items():
        if len(decisions) > 1 and len(set(decisions)) > 1:
            inconsistency_count += 1
    total_profiles = len(fairness_tracker) or 1
    consistency_score = 1.0 - (inconsistency_count / total_profiles)

    # Over-escalation penalty: escalating when hold/clear was correct
    over_escalated = [
        d for d in decision_log
        if d.get("action") == "escalate" and d.get("ground_truth") in ["clear", "hold"]
    ]
    over_escalate_rate = len(over_escalated) / len(decision_log) if decision_log else 0.0

    score = round(
        0.50 * weighted_acc
        + 0.35 * consistency_score
        - 0.15 * over_escalate_rate,
        3
    )
    score = max(0.0, min(1.0, score))

    return {
        "task_id": "task4_adversarial",
        "score": score,
        "weighted_accuracy": weighted_acc,
        "consistency_score": consistency_score,
        "over_escalation_rate": over_escalate_rate,
        "inconsistent_profiles": inconsistency_count,
        "explanation": (
            f"Weighted accuracy: {weighted_acc:.1%}, "
            f"Consistency: {consistency_score:.1%}, "
            f"Over-escalation rate: {over_escalate_rate:.1%}. "
            f"Final score: {score:.3f}"
        )
    }


# ─── Unified grader dispatcher ────────────────────────────────────────────────

def run_grader(episode_state: Dict[str, Any]) -> Dict[str, Any]:
    task_id = episode_state.get("task_id", "task1_document_check")
    log = episode_state.get("decision_log", [])
    steps = episode_state.get("step_count", 0)
    max_steps = episode_state.get("max_steps", 30)
    time_elapsed = episode_state.get("time_elapsed", 0)
    time_limit = episode_state.get("time_limit", 300)
    processed = episode_state.get("passengers_processed", 0)
    total = episode_state.get("passengers_total", 1)
    fairness = episode_state.get("fairness_tracker", {})

    if task_id == "task1_document_check":
        return grade_task1(log, steps, max_steps)
    elif task_id == "task2_flag_detection":
        return grade_task2(log, steps, max_steps)
    elif task_id == "task3_queue_pressure":
        return grade_task3(log, steps, max_steps, time_elapsed, time_limit, processed, total)
    elif task_id == "task4_adversarial":
        return grade_task4(log, steps, max_steps, fairness)
    else:
        return {"task_id": task_id, "score": 0.0, "explanation": "Unknown task."}
