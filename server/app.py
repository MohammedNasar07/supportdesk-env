from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import random
import uvicorn

from graders import grade

app = FastAPI()

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


class EnvState:
    def __init__(self):
        self.ticket = None
        self.actions: List[Dict] = []
        self.step_count = 0
        self.done = False


state = EnvState()


class Action(BaseModel):
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(task: str = "classify", seed: int = 42):
    random.seed(seed)
    state.ticket = random.choice(TICKETS)
    state.actions = []
    state.step_count = 0
    state.done = False

    return {"observation": state.ticket["body"], "done": False}


@app.post("/step")
def step(action: Action):
    if state.done:
        return {"reward": 0.0, "done": True, "observation": "", "info": {}}

    state.step_count += 1
    msg = action.message.lower()

    if "account" in msg:
        state.actions.append({"action_type": "classify", "category": "account"})
    elif "billing" in msg:
        state.actions.append({"action_type": "classify", "category": "billing"})
    elif "technical" in msg:
        state.actions.append({"action_type": "classify", "category": "technical"})

    if "critical" in msg:
        state.actions.append({"action_type": "set_priority", "priority": "critical"})
    elif "high" in msg:
        state.actions.append({"action_type": "set_priority", "priority": "high"})
    elif "medium" in msg:
        state.actions.append({"action_type": "set_priority", "priority": "medium"})

    if "support" in msg:
        state.actions.append({"action_type": "route", "team": "support"})
    elif "finance" in msg:
        state.actions.append({"action_type": "route", "team": "finance"})
    elif "engineering" in msg:
        state.actions.append({"action_type": "route", "team": "engineering"})

    if len(msg) > 20:
        state.actions.append({"action_type": "draft_response", "response_draft": msg})

    if state.step_count >= 3:
        state.done = True

        scores = grade("resolve", state.actions, state.ticket)

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


# ✅ REQUIRED MAIN FUNCTION
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


# ✅ REQUIRED ENTRY GUARD
if __name__ == "__main__":
    main()
