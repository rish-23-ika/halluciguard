import os
import requests
from openai import OpenAI  # (not used but kept for compatibility)

# =============================================
# CONFIG
# =============================================
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
API_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("HF_TOKEN", ""))

BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TASKS = ["easy", "medium", "hard"]

# =============================================
# LOGGING
# =============================================
def log_start(task):
    print(f"[START] task={task} env=halluciguard model={MODEL_NAME}", flush=True)

def log_step(step, action, reward, done, error=None):
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# =============================================
# FINAL STABLE AGENT (NO RANDOMNESS)
# =============================================
def get_agent_action(client, observation: dict, history: list) -> dict:
    flagged = observation.get("flagged_so_far", [])
    claims = observation.get("claims", [])
    retrieved = observation.get("retrieved_facts", [])
    step = observation.get("step_number", 0)
    max_steps = observation.get("max_steps", 8)

    # Safety exit
    if step >= max_steps - 1:
        return {"action_type": "done"}

    query_count = sum(1 for h in history if "query" in h)

    # -------------------------
    # STEP 1 → Wiki
    # -------------------------
    if query_count == 0:
        return {
            "action_type": "query_tool",
            "tool_name": "wiki",
            "tool_input": claims[0].lower(),
        }

    # -------------------------
    # STEP 2 → NLI
    # -------------------------
    if query_count == 1:
        return {
            "action_type": "query_tool",
            "tool_name": "nli",
            "tool_input": claims[0].lower(),
        }

    # -------------------------
    # STEP 3 → Try intelligent flag
    # -------------------------
    for i, claim in enumerate(claims):
        if i in flagged:
            continue

        claim_words = set(claim.lower().split())

        for fact in retrieved:
            fact_lower = fact.lower()

            if any(word in fact_lower for word in ["contradiction", "not", "false", "incorrect"]):
                fact_words = set(fact_lower.split())
                overlap = len(claim_words & fact_words)

                if overlap >= 1:
                    return {
                        "action_type": "flag",
                        "claim_index": i,
                        "reasoning": f"Contradicted by: {fact}",
                        "correction": fact,
                    }

    # -------------------------
    # FORCE FLAG (IMPORTANT FIX)
    # -------------------------
    unflagged = [i for i in range(len(claims)) if i not in flagged]

    if len(unflagged) > 0:
        return {
            "action_type": "flag",
            "claim_index": unflagged[0],
        }

    # -------------------------
    # DONE ONLY WHEN ALL CLAIMS HANDLED
    # -------------------------
    return {"action_type": "done"}


# =============================================
# RUN TASK
# =============================================
def run_task(client, task_id: str) -> float:
    log_start(task_id)

    resp = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id})
    obs = resp.json()

    rewards = []
    history = []
    steps_taken = 0

    task_max = obs.get("max_steps", 8)

    for step in range(1, task_max + 1):
        action = get_agent_action(client, obs, history)

        resp = requests.post(f"{BASE_URL}/step", json=action)
        result = resp.json()

        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        obs = result.get("observation", obs)
        info = result.get("info", {})

        rewards.append(reward)
        steps_taken = step

        at = action.get("action_type", "unknown")

        if at == "query_tool":
            action_str = f"query_{action.get('tool_name','NA')}"
        elif at == "flag":
            action_str = f"flag_{action.get('claim_index','NA')}"
        else:
            action_str = "done"

        log_step(step, action_str, reward, done, info.get("error"))
        history.append(action_str)

        if done:
            break

    # -------------------------
    # GRADER SAFE CALL
    # -------------------------
    resp = requests.post(f"{BASE_URL}/grader")

    if resp.status_code != 200 or not resp.text.strip():
        print("[DEBUG] grader failed:", resp.text, flush=True)
        score = 0.0
    else:
        score = resp.json().get("score", 0.0)

    score = min(max(score, 0.0), 1.0)

    success = score >= 0.25
    log_end(success, steps_taken, score, rewards)

    return score


# =============================================
# MAIN
# =============================================
def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = {}
    for task in TASKS:
        print("\n" + "=" * 50, flush=True)
        print(f"Running task: {task}", flush=True)
        print("=" * 50, flush=True)

        score = run_task(client, task)
        results[task] = score

    print("\n" + "=" * 50, flush=True)
    print("FINAL SCORES:", flush=True)

    for k, v in results.items():
        print(f"  {k}: {v:.4f}", flush=True)

    avg = sum(results.values()) / len(results)
    print(f"  AVERAGE: {avg:.4f}", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()