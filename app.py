# app.py — FINAL OpenEnv-compatible environment

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List
import random

from graders import grade

app = FastAPI()

# ───────── DATASET ─────────

TICKETS = [
    {
        "body": "I cannot login to my account",
        "gt_category": "account",
        "gt_priority": "high",
        "gt_team": "support",
        "response_keywords": ["login", "reset", "password"],
    },
    {
        "body": "Billing issue, wrong charge",
        "gt_category": "billing",
        "gt_priority": "critical",
        "gt_team": "finance",
        "response_keywords": ["refund", "billing", "charge"],
    },
    {
        "body": "Feature not working properly",
        "gt_category": "technical",
        "gt_priority": "medium",
        "gt_team": "engineering",
        "response_keywords": ["bug", "fix", "issue"],
    },
]

# ───────── STATE ─────────


class EnvState:
    def __init__(self):
        self.ticket = None
        self.actions: List[Dict] = []
        self.step_count = 0
        self.done = False


state = EnvState()


# ───────── REQUEST MODELS ─────────


class Action(BaseModel):
    message: str


# ───────── RESET ─────────


@app.post("/reset")
def reset(task: str = "classify", seed: int = 42):
    random.seed(seed)

    state.ticket = random.choice(TICKETS)
    state.actions = []
    state.step_count = 0
    state.done = False

    return {
        "observation": state.ticket["body"],
        "task": task,
        "done": False,
    }


# ───────── STEP ─────────


@app.post("/step")
def step(action: Action):
    if state.done:
        return {"reward": 0.0, "done": True, "observation": "", "info": {}}

    state.step_count += 1

    # VERY IMPORTANT: convert LLM message → fake structured actions
    msg = action.message.lower()

    parsed_action = {}

    # classify
    if "account" in msg:
        parsed_action = {"action_type": "classify", "category": "account"}
    elif "billing" in msg:
        parsed_action = {"action_type": "classify", "category": "billing"}
    elif "technical" in msg or "bug" in msg:
        parsed_action = {"action_type": "classify", "category": "technical"}

    # priority
    if "critical" in msg:
        state.actions.append({"action_type": "set_priority", "priority": "critical"})
    elif "high" in msg:
        state.actions.append({"action_type": "set_priority", "priority": "high"})
    elif "medium" in msg:
        state.actions.append({"action_type": "set_priority", "priority": "medium"})

    # team
    if "support" in msg:
        state.actions.append({"action_type": "route", "team": "support"})
    elif "finance" in msg:
        state.actions.append({"action_type": "route", "team": "finance"})
    elif "engineering" in msg:
        state.actions.append({"action_type": "route", "team": "engineering"})

    # response
    if len(msg) > 20:
        state.actions.append({"action_type": "draft_response", "response_draft": msg})

    if parsed_action:
        state.actions.append(parsed_action)

    # DONE CONDITION
    if state.step_count >= 3:
        state.done = True

        scores = grade(
            "resolve", state.actions, state.ticket  # always use hardest task
        )

        return {
            "reward": scores["total"],
            "done": True,
            "observation": "",
            "info": {"final_scores": scores},
        }

    return {
        "reward": 0.2,
        "done": False,
        "observation": state.ticket["body"],
        "info": {},
    }
