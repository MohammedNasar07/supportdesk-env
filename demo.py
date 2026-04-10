from src.env import SupportFlowEnv
from src.generator import load_tickets
from src.schemas import Ticket, AgentAction
import json

def main():
    print("🚀 SupportFlow Arena CLI Demo")
    env = SupportFlowEnv()
    tickets = load_tickets()
    
    for ticket in tickets[:3]:
        print(f"\n--- Testing Ticket: {ticket.ticket_id} ---")
        print(f"Message: {ticket.text}")
        
        # Mocking an agent's correct response
        mock_action = {
            "category": ticket.expected_category,
            "priority": ticket.expected_priority,
            "needs_clarification": ticket.ambiguous,
            "escalation": ticket.requires_escalation,
            "response": "Hello, I am investigating your issue."
        }
        
        obs, reward, done, info = env.step(mock_action)
        
        print(f"Agent Action: {json.dumps(mock_action)}")
        print(f"Final Reward: {reward}")
        print(f"Metric Breakdown: {json.dumps(info['metrics'], indent=2)}")

if __name__ == "__main__":
    main()
