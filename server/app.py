import os
import uvicorn
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
from app_ui import demo
from src.env.support_env import SupportDeskEnv
from src.env.models import TriageAction

app = FastAPI()

# Global environment instance (lazy loaded)
_env: Optional[SupportDeskEnv] = None

def get_env(task: str = "classify") -> SupportDeskEnv:
    global _env
    if _env is None or _env.task_name != task:
        _env = SupportDeskEnv(task_name=task)
    return _env

class Action(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def get_tasks():
    # Load tasks from the new config system
    from src.env.support_env import SupportDeskEnv
    temp_env = SupportDeskEnv(task_name="classify")
    strict_tasks = []
    for tid, t in temp_env.tasks.items():
        strict_tasks.append({
            "id": tid,
            "name": t.get("name", tid),
            "description": t.get("description", ""),
            "difficulty": t.get("difficulty", "medium"),
            "grader": t.get("grader", "")
        })
    return strict_tasks

@app.post("/reset")
def reset(task: str = "classify", seed: int = 42):
    env = get_env(task)
    obs = env.reset()
    # Note: TicketObservation has 'body' field, reset response needs 'observation'
    return {"observation": obs.body, "done": False}

@app.post("/step")
def step(action: Action):
    env = get_env()
    from src.env.models import ActionType
    msg = action.message.lower()
    
    action_type = ActionType.SUBMIT
    if "classify" in msg or "category" in msg:
        action_type = ActionType.CLASSIFY
    elif "priority" in msg:
        action_type = ActionType.SET_PRIORITY
    elif "route" in msg:
        action_type = ActionType.ROUTE
    elif len(msg) > 20:
        action_type = ActionType.DRAFT_RESPONSE
        
    triage_action = TriageAction(
        action_type=action_type,
        message=action.message
    )
    
    result = env.step(triage_action)
    return {
        "reward": result.reward,
        "done": result.done,
        "observation": result.observation.body,
        "info": result.info
    }

# Mount Gradio UI at the root
# We do this AFTER defining all FastAPI routes
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
