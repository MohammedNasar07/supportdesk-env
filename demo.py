import json
from src.env import SupportFlowEnv
from src.generator import load_tickets
from src.utils import format_reward

def main():
    print("🚀 SupportFlow Arena CLI Demo")
    env = SupportFlowEnv()
    tickets = load_tickets()
    
    tasks = ["classify", "triage", "resolve"]
    
    for task_name in tasks:
        # Pick a different ticket for each task demo
        ticket = tickets[tasks.index(task_name)]
        print(f"\n--- Running Task: {task_name.upper()} ---")
        print(f"Ticket ID: {ticket.ticket_id}")
        print(f"Message: {ticket.text}")
        
        # Simulating an agent's correct response
        mock_action = {
            "category": ticket.expected_category,
            "priority": ticket.expected_priority,
            "needs_clarification": ticket.ambiguous,
            "escalation": ticket.requires_escalation,
            "response": "Thank you for reaching out. We are investigating your issue."
        }
        
        # Manually reset with the specific task
        env.reset(task_name)
        env.current_ticket = ticket
        
        obs, reward, done, info = env.step(mock_action)
        
        print(f"Agent Action: {json.dumps(mock_action)}")
        print(f"Reward: {format_reward(reward)}")
        print(f"Done: {done}")
        print(f"Metric Breakdown: {json.dumps(info['metrics'], indent=2)}")

if __name__ == "__main__":
    main()
