from .schemas import Ticket
from .policy import is_security_related, is_ambiguous

SAMPLES = [
    ("T-001", "I was charged twice for my order and need this fixed today.", "billing", "high", False, False),
    ("T-002", "My account is locked and I cannot log in.", "account", "medium", False, False),
    ("T-003", "My package hasn't arrived yet and tracking hasn't updated.", "shipping", "medium", False, False),
    ("T-004", "The app crashes when I tap upload.", "technical", "medium", False, False),
    ("T-005", "Someone may have accessed my account without permission.", "security", "high", False, True),
    ("T-006", "I need help with my order.", "general", "medium", True, False),
    ("T-007", "I think my payment went through but I’m not sure.", "billing", "medium", True, False),
    ("T-008", "My order says delivered but I didn’t receive it.", "shipping", "high", False, False),
    ("T-009", "The site shows an error when I try to reset my password.", "account", "medium", False, False),
    ("T-010", "I suspect fraud on my card connected to this account.", "security", "high", False, True),
]

def build_ticket(data):
    ticket_id, msg, category, priority, ambiguous, escalation = data
    return Ticket(
        ticket_id=ticket_id,
        customer_message=msg,
        hidden_label=category,
        hidden_priority=priority,
        ambiguous=ambiguous or is_ambiguous(msg),
        escalation_required=escalation or is_security_related(msg),
    )

def generate_tickets():
    return [build_ticket(item) for item in SAMPLES]
