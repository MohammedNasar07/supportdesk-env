from pydantic import BaseModel
from typing import Optional

class Ticket(BaseModel):
    ticket_id: str
    text: str
    expected_category: str
    expected_priority: str
    ambiguous: bool
    requires_escalation: bool

class AgentAction(BaseModel):
    category: str
    priority: str
    needs_clarification: bool
    escalation: bool
    response: str
