# HalluciGuard — Tool-Augmented Hallucination Detection RL Environment

## Overview

HalluciGuard is an OpenEnv-compliant RL environment where AI agents learn to detect **and correct** factual hallucinations in LLM-generated text through **agentic tool-augmented reasoning** (HalluciChain).

Unlike traditional fact-checking approaches, agents start with **zero reference knowledge** and must:

1. **Query external tools** (Wikipedia, NLI, Knowledge Graph) to gather evidence
2. **Identify** which specific claims are hallucinated based on evidence
3. **Correct** false claims with verified facts
4. **Explain** their reasoning
5. **Manage a step budget** — every tool query and flag costs a step

## Why This Matters

Hallucination is the #1 barrier to enterprise AI deployment. Existing benchmarks give agents the answer key (reference facts) upfront. **HalluciGuard doesn't.** The agent must actively gather evidence before making decisions — modeling how real-world fact-checking actually works.

This is the first OpenEnv environment to combine **tool-augmented reasoning** with **hallucination detection RL**, directly aligned with Meta's agentic AI research direction.

## HalluciChain: How It Works

```
Agent receives passage + claims (NO reference facts)
    │
    ├─► query_tool(wiki, "topic") → retrieves factual evidence
    ├─► query_tool(nli, "claim text") → gets entailment/contradiction check
    ├─► query_tool(kg, "entity") → gets knowledge graph relations
    │
    ├─► flag(claim_index, reasoning, correction) → flags hallucinated claim
    │
    └─► done → ends episode, triggers grading
```

Three mock tools simulate real retrieval systems:
- **wiki**: Factual lookup (simulates Wikipedia/RAG retrieval)
- **nli**: Natural Language Inference (entailment/contradiction checking)
- **kg**: Knowledge Graph (structured entity relations)

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| passage | str | Text containing potential hallucinations |
| claims | List[str] | Individual claims extracted from passage |
| flagged_so_far | List[int] | Already-flagged claim indices |
| retrieved_facts | List[str] | Facts gathered via tool queries (starts empty) |
| step_number | int | Current step |
| task_id | str | easy / medium / hard |
| max_steps | int | Step budget |
| tools_available | List[str] | ["wiki", "nli", "kg"] |

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| action_type | str | "query_tool", "flag", or "done" |
| tool_name | str (optional) | "wiki", "nli", or "kg" |
| tool_input | str (optional) | Query string for the tool |
| claim_index | int (optional) | Which claim to flag |
| reasoning | str (optional) | Why it's hallucinated |
| correction | str (optional) | Corrected version of the claim |

## Reward Function

| Action | Reward |
|--------|--------|
| Useful tool query | +0.05 |
| Redundant/irrelevant query | -0.02 |
| Evidence-backed correct flag | +0.15 |
| Blind correct flag (no evidence) | +0.05 |
| Valid reasoning | +0.10 |
| Valid correction | +0.30 |
| Wrong flag (false positive) | -0.25 |
| Repeated flag | -0.20 |
| All hallucinations found + done | +0.20 |
| Missed hallucination | -0.10 each |

## Grading

F1 score + reasoning bonus (up to +0.10) + tool usage bonus (up to +0.10) - efficiency penalty (0.03 per step). Score range: 0.0 to 1.0.

## Tasks

| Difficulty | Description | Passages | Max Steps |
|-----------|-------------|----------|-----------|
| Easy | Obvious contradictions, simple tool lookups | 10 | 8 |
| Medium | Subtle numerical/temporal errors | 10 | 8 |
| Hard | Multi-hop reasoning, combining multiple facts | 5 | 10 |

**Total: 25 evaluation scenarios** (including clean passages with zero hallucinations)

## Baseline Scores

| Task | Expected Range |
|------|---------------|
| Easy | 0.25 - 0.50 |
| Medium | 0.20 - 0.45 |
| Hard | 0.15 - 0.40 |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Health check |
| /reset | POST | Start new episode |
| /step | POST | Submit action |
| /state | GET | Full environment state |
| /tasks | GET | List available tasks |
| /grader | POST | Get episode F1 score |
| /analysis | GET | Detailed behavioral breakdown |
| /metadata | GET | Environment metadata |
| /schema | GET | Observation/Action schemas |

## Tech Stack

- **Framework**: FastAPI + Pydantic (typed models, OpenEnv-compliant)
- **Tool Simulation**: Mock RAG/NLI/KG via JSON-defined tool responses (no external dependencies)
- **Grading**: Deterministic F1-based grader with bonus/penalty system
- **Inference**: OpenAI-compatible client (works with Groq, HuggingFace, any provider)
- **Deployment**: Docker container on HuggingFace Spaces

## Setup

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Docker

```bash
docker build -t halluciguard .
docker run -p 7860:7860 halluciguard
```

## Inference

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="your-key"
python inference.py
```

## Team

- Rishika Rastogi (Lead)
- Aadi Jangir
- Tanish Vashistha
