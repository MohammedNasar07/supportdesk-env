import os
import json
import uvicorn
import gradio as gr
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional

from src.env import SupportFlowEnv
from src.generator import load_tickets
from src.schemas import AgentAction
from src.grader import grade_episode

# ─── FASTAPI BACKEND ───
app = FastAPI()
env = SupportFlowEnv()
tickets = load_tickets()

class Action(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def get_tasks():
    return [
        {
            "id": "triage",
            "name": "Support Triage",
            "description": "Categorize and prioritize customer support tickets.",
            "difficulty": "medium",
            "grader": "src.grader:grade_episode"
        }
    ]

@app.post("/reset")
def reset(task: str = "triage", seed: int = 42):
    obs = env.reset(task)
    return {"observation": obs, "done": False}

@app.post("/step")
def step(action: Action):
    try:
        # Evaluator sends JSON string in 'message'
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
def run_agent_demo(ticket_id):
    ticket = next((t for t in tickets if t.ticket_id == ticket_id), tickets[0])
    # Mock response for demo
    mock_action = {
        "category": ticket.expected_category,
        "priority": ticket.expected_priority,
        "needs_clarification": ticket.ambiguous,
        "escalation": ticket.requires_escalation,
        "response": f"Thanks for your message regarding {ticket.expected_category}. We are looking into it."
    }
    agent_action = AgentAction(**mock_action)
    result = grade_episode(ticket, agent_action)
    return ticket.text, json.dumps(mock_action, indent=2), json.dumps(result, indent=2)

with gr.Blocks() as demo:
    gr.Markdown("# SupportFlow Arena: Triage Dashboard")
    gr.Markdown("Deterministic RL-style support environment. Select a ticket to see how the agent and grader perform.")
    with gr.Row():
        ticket_dropdown = gr.Dropdown(choices=[t.ticket_id for t in tickets], label="Select Ticket ID", value=tickets[0].ticket_id)
        run_btn = gr.Button("Run Agent & Grade")
    with gr.Row():
        ticket_text = gr.Textbox(label="Ticket Text", lines=4)
    with gr.Row():
        agent_out = gr.Code(label="Agent JSON Output", language="json")
        grader_out = gr.Code(label="Grader Scoring Breakdown", language="json")
    run_btn.click(run_agent_demo, inputs=[ticket_dropdown], outputs=[ticket_text, agent_out, grader_out])

# ─── MOUNT & RUN ───
# Mount Gradio into FastAPI at root (/)
# This allows both API calls (POST /reset) and UI access
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    # We use uvicorn to serve the combined app
    uvicorn.run(app, host="0.0.0.0", port=port)
