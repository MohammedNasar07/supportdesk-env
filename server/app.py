import os
import uvicorn
import json
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from src.env import SupportFlowEnv
from inference import infer_action

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

# ───────── GRADIO UI ─────────
def run_ui_demo(ticket_text):
    # This is for the human-facing UI
    from src.generator import build_ticket
    from src.grader import grade
    from src.schemas import AgentAction
    
    ticket = build_ticket(("T-DEMO", ticket_text, "general", "medium", False, False))
    action_dict = infer_action(ticket_text)
    action = AgentAction(**action_dict)
    
    # Heuristic labels for UI scoring
    if "charged" in ticket_text.lower() or "payment" in ticket_text.lower():
        ticket.hidden_label = "billing"
    elif "login" in ticket_text.lower() or "password" in ticket_text.lower():
        ticket.hidden_label = "account"
    
    result = grade(ticket, action)
    return json.dumps(action_dict, indent=2), json.dumps(result, indent=2)

with gr.Blocks() as demo:
    gr.Markdown("# SupportFlow Arena")
    gr.Markdown("An OpenEnv-style customer support environment. Paste a ticket, get a decision, and see the score.")
    t_in = gr.Textbox(label="Ticket Text", lines=5)
    btn = gr.Button("Run Agent")
    out = gr.Code(label="Agent Output", language="json")
    sc = gr.Code(label="Scores", language="json")
    btn.click(run_ui_demo, inputs=t_in, outputs=[out, sc])

# Mount Gradio into FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
