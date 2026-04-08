from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# ==========================
# ACTION
# ==========================
class Action(BaseModel):
    action_type: str = Field(..., description="query_tool | flag | done")
    tool_name: Optional[str] = Field(None, description="wiki | nli | kg")
    tool_input: Optional[str] = Field(None, description="query string")
    claim_index: Optional[int] = Field(None, description="claim index to flag")
    reasoning: Optional[str] = Field(None, description="reason for flag")
    correction: Optional[str] = Field(None, description="corrected claim")


# ==========================
# OBSERVATION
# ==========================
class Observation(BaseModel):
    passage: str
    claims: List[str]
    flagged_so_far: List[int]
    retrieved_facts: List[str]
    step_number: int
    task_id: str
    max_steps: int
    tools_available: List[str] = ["wiki", "nli", "kg"]


# ==========================
# ENV STATE (INTERNAL)
# ==========================
class EnvironmentState(BaseModel):
    current_passage: str
    claims: List[str]
    ground_truth_hallucinations: List[int]

    agent_flags: List[int]
    retrieved_facts: List[str]

    reasoning_count: int
    tool_queries_made: int
    step_count: int
    cumulative_reward: float

    done: bool
    task_id: str
    max_steps: int

    # hidden/internal
    reference_facts: List[str]
    tool_responses: Dict[str, str]   # ✅ FIXED


# ==========================
# STEP RESPONSE
# ==========================
class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ==========================
# TASK INFO
# ==========================
class TaskInfo(BaseModel):
    task_id: str
    difficulty: str
    description: str
    num_passages: int


# ==========================
# GRADER RESPONSE
# ==========================
class GraderResponse(BaseModel):
    task_id: str
    score: float
    details: Dict[str, Any] = Field(default_factory=dict)