from .schemas import Ticket, AgentAction
from .policy import has_forbidden_promise

def score_category(pred: str, gold: str) -> float:
    return 1.0 if pred == gold else 0.0

def score_priority(pred: str, gold: str) -> float:
    return 1.0 if pred == gold else 0.0

def score_clarification(action: AgentAction, ticket: Ticket) -> float:
    if ticket.ambiguous and action.needs_clarification:
        return 1.0
    if ticket.ambiguous and not action.needs_clarification:
        return 0.0
    if not ticket.ambiguous and action.needs_clarification:
        return 0.5
    return 1.0

def score_escalation(action: AgentAction, ticket: Ticket) -> float:
    return 1.0 if action.escalation == ticket.escalation_required else 0.0

def score_policy(action: AgentAction) -> float:
    if has_forbidden_promise(action.response):
        return 0.0
    return 1.0

def score_response(action: AgentAction) -> float:
    text = action.response.strip()
    if len(text) < 15:
        return 0.2
    if "sorry" in text.lower() or "help" in text.lower():
        return 1.0
    return 0.7

def grade(ticket: Ticket, action: AgentAction) -> dict:
    c = score_category(action.category, ticket.hidden_label)
    p = score_priority(action.priority, ticket.hidden_priority)
    cl = score_clarification(action, ticket)
    e = score_escalation(action, ticket)
    pol = score_policy(action)
    r = score_response(action)

    total = round(
        0.30 * c +
        0.20 * p +
        0.20 * cl +
        0.20 * e +
        0.10 * r,
        3
    )

    return {
        "ticket_id": ticket.ticket_id,
        "category_score": c,
        "priority_score": p,
        "clarification_score": cl,
        "escalation_score": e,
        "policy_score": pol,
        "response_score": r,
        "total_score": total,
    }
