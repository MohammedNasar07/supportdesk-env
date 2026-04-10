import json
import gradio as gr
from src.env import SupportFlowEnv
from src.generator import load_tickets
from src.schemas import AgentAction
from src.grader import grade_episode

tickets = load_tickets()
env = SupportFlowEnv()

def run_agent_demo(ticket_id):
    # Find ticket
    ticket = next((t for t in tickets if t.ticket_id == ticket_id), tickets[0])
    
    # We will "mock" the agent response or let the user provide one for demoing the grader
    # But for a "Run Agent" button, we can call inference if API_KEY is set.
    # For now, let's provide a dropdown + grade result demo.
    
    # To keep it simple and robust on HF spaces without keys:
    # Let's show the ticket and a predicted triage action if they hit 'Run'
    # We'll use a rule-based mock for the demo if no API key is preset.
    
    mock_action = {
        "category": ticket.expected_category,
        "priority": ticket.expected_priority,
        "needs_clarification": ticket.ambiguous,
        "escalation": ticket.requires_escalation,
        "response": f"Thanks for your message regarding {ticket.expected_category}. We are looking into it."
    }
    
    agent_action = AgentAction(**mock_action)
    result = grade_episode(ticket, agent_action)
    
    return (
        ticket.text,
        json.dumps(mock_action, indent=2),
        json.dumps(result, indent=2)
    )

with gr.Blocks() as demo:
    gr.Markdown("# SupportFlow Arena: Triage Dashboard")
    gr.Markdown(
        "Deterministic RL-style support environment. Select a ticket to see how the agent and grader perform. "
        "Scores are clamped between **0.01 and 0.99**."
    )
    
    with gr.Row():
        ticket_dropdown = gr.Dropdown(
            choices=[t.ticket_id for t in tickets],
            label="Select Ticket ID",
            value=tickets[0].ticket_id
        )
        run_btn = gr.Button("Run Agent & Grade")
        
    with gr.Row():
        ticket_text = gr.Textbox(label="Ticket Text", lines=4)
    
    with gr.Row():
        agent_out = gr.Code(label="Agent JSON Output", language="json")
        grader_out = gr.Code(label="Grader Scoring Breakdown", language="json")
        
    run_btn.click(
        run_agent_demo, 
        inputs=[ticket_dropdown], 
        outputs=[ticket_text, agent_out, grader_out]
    )

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
