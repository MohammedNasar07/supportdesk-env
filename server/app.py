import os
import uvicorn
import json
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional

from src.env import SupportFlowEnv

app = FastAPI()

# Global environment instance
_env = SupportFlowEnv()

class Action(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def get_tasks():
    # Return tasks compatible with OpenEnv schema
    return [
        {
            "id": "classify",
            "name": "Ticket Categorization",
            "description": "Categorize, prioritize, and triage support tickets.",
            "difficulty": "medium",
            "grader": "src.grader:grade"
        }
    ]

@app.post("/reset")
def reset(task: str = "classify", seed: int = 42):
    ticket = _env.reset()
    return {
        "observation": ticket.customer_message,
        "done": False,
        "info": {"ticket_id": ticket.ticket_id}
    }

@app.post("/step")
def step(action: Action):
    # The evaluator sends JSON string in 'message'
    try:
        action_dict = json.loads(action.message)
    except:
        action_dict = {"category": "general", "priority": "medium", "needs_clarification": True, "escalation": False, "response": action.message}
    
    next_obs, reward, done = _env.step(action_dict)
    
    obs_text = next_obs.customer_message if next_obs else ""
    
    return {
        "reward": reward["total_score"],
        "done": done,
        "observation": obs_text,
        "info": reward
    }

def main():
    import gradio as gr
    from app import demo
    
    # Mount Gradio UI
    # Note: Using '/ui' for Gradio and root for API is safest
    combined_app = gr.mount_gradio_app(app, demo, path="/")
    
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(combined_app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
