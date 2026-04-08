import random
from typing import Tuple, Dict, Any, Optional
from models import Observation, Action, EnvironmentState
from graders import compute_f1_score


class HalluciGuardEnv:
    def __init__(self, tasks: dict):
        self.tasks = tasks
        self.state: Optional[EnvironmentState] = None

    def reset(self, task_id: str = None) -> Observation:
        if task_id is None:
            task_id = list(self.tasks.keys())[0]

        if task_id not in self.tasks:
            raise ValueError(f"Unknown task_id: {task_id}")

        task = random.choice(self.tasks[task_id])

        self.state = EnvironmentState(
            current_passage=task.get("passage", ""),
            claims=task.get("claims", []),
            ground_truth_hallucinations=task.get("hallucinations", []),
            agent_flags=[],
            retrieved_facts=[],
            reasoning_count=0,
            tool_queries_made=0,
            step_count=0,
            cumulative_reward=0.0,
            done=False,
            task_id=task_id,
            max_steps=task.get("max_steps", 8),
            reference_facts=task.get("reference_facts", []),
            tool_responses=task.get("tool_responses", {}),
        )

        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() first")

        if self.state.done:
            return self._get_observation(), 0.0, True, {"error": "Episode done"}

        reward = 0.0
        info: Dict[str, Any] = {}
        self.state.step_count += 1

        if action.action_type == "query_tool":
            tool = (action.tool_name or "").lower()
            query = (action.tool_input or "").lower()

            self.state.tool_queries_made += 1

            found = False
            for key, val in self.state.tool_responses.items():
                tool_prefix, _, key_text = key.partition(":")

                if tool_prefix != tool:
                    continue

                if self._fuzzy_match(query, key_text):
                    fact_text = str(val)
                    if fact_text not in self.state.retrieved_facts:
                        self.state.retrieved_facts.append(fact_text)
                        reward += 0.05
                        info["result"] = "fact_retrieved"
                    else:
                        reward -= 0.02
                        info["result"] = "duplicate"
                    found = True
                    break

            if not found:
                reward -= 0.02
                info["error"] = "no_tool_match"

        elif action.action_type == "flag":
            idx = action.claim_index

            if idx is None or idx < 0 or idx >= len(self.state.claims):
                return self._penalty(-0.1, "invalid index")

            if idx in self.state.agent_flags:
                return self._penalty(-0.2, "already flagged")

            self.state.agent_flags.append(idx)

            if idx in self.state.ground_truth_hallucinations:
                reward += 0.15 if self.state.retrieved_facts else 0.05

                if action.reasoning:
                    reward += 0.10
                    self.state.reasoning_count += 1

                if action.correction and self._is_correction_valid(action.correction):
                    reward += 0.30

                info["result"] = "correct_flag"
            else:
                reward -= 0.25
                info["result"] = "wrong_flag"

        elif action.action_type == "done":
            self.state.done = True

            pred = set(self.state.agent_flags)
            truth = set(self.state.ground_truth_hallucinations)
            missed = len(truth - pred)

            if missed == 0:
                reward += 0.20
            else:
                reward -= 0.10 * missed

        else:
            return self._penalty(-0.1, "invalid action")

        if self.state.step_count >= self.state.max_steps:
            self.state.done = True

        self.state.cumulative_reward += reward

        return self._get_observation(), round(reward, 4), self.state.done, info

    def _fuzzy_match(self, query: str, key: str) -> bool:
        stop_words = {"is", "the", "of", "in", "and", "a", "to", "at", "on", "for"}
        query_words = set(query.lower().split()) - stop_words
        key_words = set(key.lower().replace(":", " ").split()) - stop_words
        return len(query_words & key_words) >= 1

    def _is_correction_valid(self, correction: str) -> bool:
        correction = correction.lower()
        for fact in self.state.reference_facts:
            overlap = sum(1 for w in correction.split() if w in fact.lower())
            if overlap >= 3:
                return True
        return False

    def _penalty(self, val, msg):
        self.state.cumulative_reward += val
        return self._get_observation(), val, False, {"error": msg}

    def get_state(self) -> Optional[EnvironmentState]:
        return self.state

    def get_final_score(self) -> float:
        if not self.state:
            return 0.0
        try:
            return compute_f1_score(
                self.state.agent_flags,
                self.state.ground_truth_hallucinations,
                reasoning_count=self.state.reasoning_count,
                step_count=self.state.step_count,
                tool_queries=self.state.tool_queries_made,
                facts_retrieved=len(self.state.retrieved_facts),
            )
        except Exception:
            return 0.0

    def _get_observation(self) -> Observation:
        return Observation(
            passage=self.state.current_passage,
            claims=self.state.claims,
            flagged_so_far=list(self.state.agent_flags),
            retrieved_facts=list(self.state.retrieved_facts),
            step_number=self.state.step_count,
            task_id=self.state.task_id,
            max_steps=self.state.max_steps,
            tools_available=["wiki", "nli", "kg"],
        )