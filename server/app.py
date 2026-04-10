import os
import json
import uvicorn
import gradio as gr
from fastapi import FastAPI, Body
from pydantic import BaseModel
import sys
from typing import Optional

# Ensure the root directory is in the Python path for 'src' imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env import SupportFlowEnv
from src.generator import load_tickets
from src.schemas import AgentAction
from src.grader import grade_episode

# ─── FASTAPI BACKEND ───
app = FastAPI()
env = SupportFlowEnv()
tickets = load_tickets()
if not tickets:
    print("WARNING: No tickets loaded! Creating a dummy ticket.", file=sys.stderr)
    from src.schemas import Ticket
    tickets = [Ticket(
        ticket_id="DUMMY-01",
        text="Sample message.",
        expected_category="general",
        expected_priority="low",
        ambiguous=True,
        requires_escalation=False
    )]

class Action(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def get_tasks():
    # MANDATORY: At least 3 tasks to pass Phase 2 Task Validation
    return [
        {
            "id": "classify",
            "name": "Ticket Classification",
            "description": "Determine category and priority.",
            "difficulty": "easy",
            "grader": "src.grader:grade_classify"
        },
        {
            "id": "triage",
            "name": "Triage Decider",
            "description": "Handle escalation and ambiguity.",
            "difficulty": "medium",
            "grader": "src.grader:grade_triage"
        },
        {
            "id": "resolve",
            "name": "Full Issue Resolution",
            "description": "Compose policy-safe responses.",
            "difficulty": "hard",
            "grader": "src.grader:grade_resolve"
        }
    ]

@app.post("/reset")
def reset(task: str = "triage", seed: int = 42):
    obs = env.reset(task)
    return {"observation": obs, "done": False}

@app.post("/step")
def step(action: Action):
    try:
        action_dict = json.loads(action.message)
    except:
        action_dict = {
            "category": "general",
            "priority": "low",
            "needs_clarification": True,
            "escalation": False,
            "response": action.message
        }
    
    obs, reward, done, info = env.step(action_dict)
    return {
        "reward": reward,
        "done": done,
        "observation": obs,
        "info": info
    }

# ─── GRADIO FRONTEND ───
def run_agent_demo(ticket_id, task_id):
    ticket = next((t for t in tickets if t.ticket_id == ticket_id), tickets[0])
    response_draft = f"I've analyzed your {ticket.expected_category} issue. We're on it."
    
    mock_action = {
        "category": ticket.expected_category,
        "priority": ticket.expected_priority,
        "needs_clarification": ticket.ambiguous,
        "escalation": ticket.requires_escalation,
        "response": response_draft
    }
    
    agent_action = AgentAction(**mock_action)
    result = grade_episode(ticket, agent_action, task=task_id)
    return ticket.text, json.dumps(mock_action, indent=2), json.dumps(result, indent=2)

with gr.Blocks() as demo:
    gr.Markdown("# SupportFlow Arena: Triage Dashboard")
    gr.Markdown("Deterministic RL-style support environment with multiple tasks and mandatory high-density gradients.")
    
    with gr.Row():
        t_id = gr.Dropdown(choices=[t.ticket_id for t in tickets], label="Ticket ID", value=tickets[0].ticket_id)
        task_choice = gr.Dropdown(choices=["classify", "triage", "resolve"], label="Task Type", value="triage")
        run_btn = gr.Button("Evaluate Agent")
        
    with gr.Row():
        t_text = gr.Textbox(label="Ticket Message", lines=4)
        
    with gr.Row():
        agent_out = gr.Code(label="Agent Generated Decision", language="json")
        score_out = gr.Code(label="Score Breakdown (Clamped 0.01-0.99)", language="json")
        
    run_btn.click(run_agent_demo, inputs=[t_id, task_choice], outputs=[t_text, agent_out, score_out])

app = gr.mount_gradio_app(app, demo, path="/")

def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
