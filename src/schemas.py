from typing import Literal, Optional
from pydantic import BaseModel, Field

Category = Literal["billing", "account", "shipping", "technical", "security", "general"]
Priority = Literal["low", "medium", "high"]

class Ticket(BaseModel):
    ticket_id: str
    customer_message: str
    customer_tier: str = "standard"
    history: list[str] = Field(default_factory=list)
    hidden_label: Category = "general"
    hidden_priority: Priority = "medium"
    ambiguous: bool = False
    escalation_required: bool = False

class AgentAction(BaseModel):
    category: Category
    priority: Priority
    needs_clarification: bool
    escalation: bool
    response: str
