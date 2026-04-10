import random
from typing import Tuple, Dict, Any, List
from .schemas import Ticket, AgentAction
from .generator import load_tickets
from .grader import grade_episode

class SupportFlowEnv:
    def __init__(self, tickets_file: str = None):
        self.tickets = load_tickets(tickets_file)
        self.current_ticket = None
        self.task_name = None

    def reset(self, task_name: str = "classify") -> str:
        """
        Reset the environment and return the ticket text.
        """
        self.task_name = task_name
        self.current_ticket = random.choice(self.tickets)
        return self.current_ticket.text

    def step(self, action_dict: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Evaluate the action and return (observation, reward, done, info).
        """
        try:
            # Validate input using AgentAction schema
            agent_action = AgentAction(**action_dict)
        except Exception:
            # Fallback for invalid formats
            agent_action = AgentAction(
                category="general",
                priority="low",
                needs_clarification=True,
                escalation=False,
                response="I'm sorry, I encountered an internal error processing your request."
            )
            
        result = grade_episode(self.current_ticket, agent_action)
        
        reward = result["total_score"]
        done = True  # Deterministic single-step episodes for triage
        observation = "Episode Finished"
        
        info = {
            "ticket_id": self.current_ticket.ticket_id,
            "metrics": result
        }
        
        return observation, reward, done, info
