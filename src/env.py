import random
from typing import Tuple, Dict, Any, List
from .schemas import Ticket, AgentAction
from .generator import load_tickets
from .grader import grade_episode

class SupportFlowEnv:
    def __init__(self, tickets_file: str = None):
        self.tickets = load_tickets(tickets_file)
        self.current_ticket = None
        self.task_name = "triage"

    def reset(self, task_name: str = "triage") -> str:
        """
        Reset the environment and return the ticket text.
        """
        self.task_name = task_name
        self.current_ticket = random.choice(self.tickets)
        return self.current_ticket.text

    def step(self, action_dict: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Evaluate the action using the correct task grader.
        """
        try:
            agent_action = AgentAction(**action_dict)
        except Exception:
            agent_action = AgentAction(
                category="general",
                priority="low",
                needs_clarification=True,
                escalation=False,
                response="Error processing request."
            )
            
        try:
            if not self.current_ticket:
                self.reset(self.task_name)
            result = grade_episode(self.current_ticket, agent_action, self.task_name)
        except Exception as e:
            result = {"total_score": 0.01, "error": str(e)}
            
        reward = result.get("score", result.get("total_score", 0.01))
        reward = round(max(0.01, min(0.99, float(reward))), 4)
        done = True
        observation = "Task Finished"
        
        info = {
            "ticket_id": self.current_ticket.ticket_id if self.current_ticket else "unknown",
            "task": self.task_name,
            "metrics": result
        }
        
        return observation, reward, done, info
