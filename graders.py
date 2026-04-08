from typing import List


def compute_f1_score(
    agent_flags: List[int],
    ground_truth: List[int],
    reasoning_count: int = 0,
    step_count: int = 8,
    tool_queries: int = 0,
    facts_retrieved: int = 0,
) -> float:
    """
    Grader: F1 + reasoning bonus + tool usage bonus - efficiency penalty.
    Returns score in [0.0, 1.0].
    """
    agent_set = set(agent_flags)
    truth_set = set(ground_truth)

    # handle clean passages (no hallucinations)
    if len(truth_set) == 0 and len(agent_set) == 0:
        base = 0.8  # good but not perfect — still need tool usage
    elif len(truth_set) == 0 and len(agent_set) > 0:
        return max(0.0, 0.2 - 0.1 * len(agent_set))  # penalize false flags
    elif len(agent_set) == 0:
        return 0.0
    else:
        correct = len(agent_set & truth_set)
        precision = correct / len(agent_set)
        recall = correct / len(truth_set)
        if precision + recall == 0:
            base = 0.0
        else:
            base = 2 * (precision * recall) / (precision + recall)

    # reasoning bonus: up to 0.10
    reasoning_bonus = min(0.10, reasoning_count * 0.04)

    # tool usage bonus: reward agents that gather evidence
    tool_bonus = min(0.10, facts_retrieved * 0.03)

    # efficiency penalty: more steps = lower score
    efficiency_penalty = min(0.25, 0.03 * step_count)

    final = base + reasoning_bonus + tool_bonus - efficiency_penalty
    return min(1.0, max(0.0, round(final, 4)))