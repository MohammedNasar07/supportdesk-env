from .generator import generate_tickets
from .grader import grade
from .schemas import AgentAction

class SupportFlowEnv:
    def __init__(self):
        self.tickets = generate_tickets()
        self.index = 0

    def reset(self):
        self.index = 0
        return self.tickets[self.index]

    def step(self, action_dict):
        ticket = self.tickets[self.index]
        action = AgentAction(**action_dict)
        reward = grade(ticket, action)

        done = self.index >= len(self.tickets) - 1
        if not done:
            self.index += 1
            next_obs = self.tickets[self.index]
        else:
            next_obs = None

        return next_obs, reward, done
