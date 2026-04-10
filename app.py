import os
import json
import gradio as gr
from inference import run_inference

def triage_ticket(ticket_text):
    try:
        result = run_inference(ticket_text)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# Define common examples for judges
EXAMPLES = [
    ["I was charged twice for my subscription and need this fixed today."],
    ["My account says locked, but I don't know why."],
    ["Someone may have accessed my account without permission."],
]

demo = gr.Interface(
    fn=triage_ticket,
    inputs=gr.Textbox(
        lines=6,
        label="Customer Support Ticket",
        placeholder="Paste a customer ticket here..."
    ),
    outputs=gr.Code(label="SupportFlow Agent Decision (JSON)", language="json"),
    examples=EXAMPLES,
    title="SupportFlow Arena",
    description=(
        "### 🏆 Winner-Ready OpenEnv Environment\n"
        "Benchmark your AI support agents. It classifies tickets, assigns priority, "
        "detects security risks, and generates policy-safe responses."
    ),
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
