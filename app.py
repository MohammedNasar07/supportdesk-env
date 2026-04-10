import json
import gradio as gr

from inference import infer_action
from src.generator import build_ticket
from src.grader import grade
from src.schemas import AgentAction

def run_demo(ticket_text):
    ticket = build_ticket(("T-DEMO", ticket_text, "general", "medium", False, False))

    action_dict = infer_action(ticket_text)
    action = AgentAction(**action_dict)

    # For demo, derive hidden labels from text heuristically
    # so the UI still shows a score.
    if "charged" in ticket_text.lower() or "payment" in ticket_text.lower():
        ticket.hidden_label = "billing"
        ticket.hidden_priority = "high" if "twice" in ticket_text.lower() else "medium"
    elif "login" in ticket_text.lower() or "password" in ticket_text.lower() or "account" in ticket_text.lower():
        ticket.hidden_label = "account"
        ticket.hidden_priority = "medium"
    elif "shipping" in ticket_text.lower() or "package" in ticket_text.lower() or "delivered" in ticket_text.lower():
        ticket.hidden_label = "shipping"
        ticket.hidden_priority = "medium"
    elif "crash" in ticket_text.lower() or "error" in ticket_text.lower() or "bug" in ticket_text.lower():
        ticket.hidden_label = "technical"
        ticket.hidden_priority = "medium"
    elif any(k in ticket_text.lower() for k in ["hack", "fraud", "stolen", "unauthorized", "compromised"]):
        ticket.hidden_label = "security"
        ticket.hidden_priority = "high"
        ticket.escalation_required = True

    ticket.ambiguous = any(k in ticket_text.lower() for k in ["not sure", "maybe", "i think", "unclear", "don't know"])
    result = grade(ticket, action)

    return (
        json.dumps(action_dict, indent=2),
        json.dumps(result, indent=2),
    )

with gr.Blocks() as demo:
    gr.Markdown("# SupportFlow Arena")
    gr.Markdown(
        "An OpenEnv-style customer support triage environment. "
        "Paste a ticket, get an agent decision, and see the score."
    )

    ticket_input = gr.Textbox(
        label="Customer Support Ticket",
        lines=5,
        placeholder="Example: I was charged twice for my order and need this fixed today."
    )

    run_button = gr.Button("Run Agent")
    output_json = gr.Code(label="Agent Output (JSON)", language="json")
    score_json = gr.Code(label="Grader Result", language="json")

    run_button.click(run_demo, inputs=ticket_input, outputs=[output_json, score_json])

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
