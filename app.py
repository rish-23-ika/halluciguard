from fastapi import FastAPI, HTTPException
from models import Action, StepResponse, TaskInfo, GraderResponse, Observation, EnvironmentState
from env import HalluciGuardEnv
import json


app = FastAPI(
    title="HalluciGuard",
    description="AI Hallucination Detection & Correction Environment with Tool-Augmented Reasoning — OpenEnv Compliant",
    version="2.0.0",
)


# ==========================
# LOAD TASKS
# ==========================
def load_tasks() -> dict:
    tasks = {}
    for difficulty in ["easy", "medium", "hard"]:
        try:
            with open(f"tasks/{difficulty}.json") as f:
                tasks[difficulty] = json.load(f)
        except FileNotFoundError:
            tasks[difficulty] = []
    return tasks


tasks_data = load_tasks()
env = HalluciGuardEnv(tasks_data)


# ==========================
# CORE ENDPOINTS
# ==========================

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(task_id: str = None):
    try:
        return env.reset(task_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(action: Action):
    try:
        obs, reward, done, info = env.step(action)
        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info=info
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    st = env.get_state()
    if st is None:
        return {"error": "env not initialized"}
    return st


@app.get("/tasks")
def list_tasks():
    result = []
    for difficulty in ["easy", "medium", "hard"]:
        result.append(TaskInfo(
            task_id=difficulty,
            difficulty=difficulty,
            description=f"HalluciChain detection — {difficulty}",
            num_passages=len(tasks_data.get(difficulty, [])),
        ))
    return result


# ==========================
# 🔥 FIXED GRADER
# ==========================
@app.post("/grader")
def grader():
    st = env.get_state()

    # ✅ CRASH FIX
    if st is None:
        return GraderResponse(
            task_id="unknown",
            score=0.0,
            details={"error": "env not initialized"}
        )

    score = env.get_final_score()

    return GraderResponse(
        task_id=st.task_id,
        score=score,
        details={
            "agent_flags": st.agent_flags,
            "ground_truth": st.ground_truth_hallucinations,
            "reasoning_count": st.reasoning_count,
            "tool_queries": st.tool_queries_made,
            "facts_retrieved": len(st.retrieved_facts),
            "steps_taken": st.step_count,
        },
    )


# ==========================
# OPTIONAL ANALYSIS (SAFE)
# ==========================
@app.get("/analysis")
def analysis():
    st = env.get_state()
    if st is None:
        return {"error": "env not initialized"}

    return {
        "flags": st.agent_flags,
        "truth": st.ground_truth_hallucinations,
        "facts": st.retrieved_facts,
        "steps": st.step_count
    }


# ==========================
# REQUIRED OPENENV ENDPOINTS
# ==========================

@app.get("/metadata")
def metadata():
    return {
        "name": "halluciguard",
        "description": "AI Hallucination Detection with Tool-Augmented Reasoning (HalluciChain)",
        "version": "2.0.0"
    }


@app.get("/schema")
def schema():
    return {
        "observation": Observation.model_json_schema(),
        "action": Action.model_json_schema(),
        "state": EnvironmentState.model_json_schema()
    }


@app.post("/mcp")
def mcp():
    return {
        "jsonrpc": "2.0",
        "result": "ok",
        "id": 1
    }


# ==========================
# MAIN
# ==========================
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()