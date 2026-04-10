from typing import Any
from .schemas import Ticket, AgentAction
from .policy import policy_check

def clamp(score: float) -> float:
    """Strictly enforce (0, 1) range to avoid validator failures."""
    return round(max(0.01, min(0.99, float(score))), 4)

def _safe_ticket(ticket: Any) -> Ticket:
    """Safely coerce any input to a Ticket object."""
    if isinstance(ticket, Ticket):
        return ticket
    if isinstance(ticket, dict):
        return Ticket(**{
            "ticket_id": ticket.get("ticket_id", "unknown"),
            "text": ticket.get("text", ""),
            "expected_category": ticket.get("expected_category", "general"),
            "expected_priority": ticket.get("expected_priority", "low"),
            "ambiguous": bool(ticket.get("ambiguous", False)),
            "requires_escalation": bool(ticket.get("requires_escalation", False)),
        })
    # If it's some other object, try to read attributes
    try:
        return Ticket(
            ticket_id=getattr(ticket, "ticket_id", "unknown"),
            text=getattr(ticket, "text", ""),
            expected_category=getattr(ticket, "expected_category", "general"),
            expected_priority=getattr(ticket, "expected_priority", "low"),
            ambiguous=bool(getattr(ticket, "ambiguous", False)),
            requires_escalation=bool(getattr(ticket, "requires_escalation", False)),
        )
    except Exception:
        return Ticket(ticket_id="fallback", text="", expected_category="general",
                      expected_priority="low", ambiguous=False, requires_escalation=False)

def _safe_action(action: Any) -> AgentAction:
    """Safely coerce any input to an AgentAction object."""
    if isinstance(action, AgentAction):
        return action
    if isinstance(action, dict):
        return AgentAction(**{
            "category": action.get("category", "general"),
            "priority": action.get("priority", "low"),
            "needs_clarification": bool(action.get("needs_clarification", False)),
            "escalation": bool(action.get("escalation", False)),
            "response": action.get("response", ""),
        })
    try:
        return AgentAction(
            category=getattr(action, "category", "general"),
            priority=getattr(action, "priority", "low"),
            needs_clarification=bool(getattr(action, "needs_clarification", False)),
            escalation=bool(getattr(action, "escalation", False)),
            response=getattr(action, "response", ""),
        )
    except Exception:
        return AgentAction(category="general", priority="low",
                           needs_clarification=False, escalation=False, response="")

def grade_classify(ticket: Any, action: Any) -> float:
    try:
        t = _safe_ticket(ticket)
        a = _safe_action(action)
        c = 0.99 if a.category.lower() == t.expected_category.lower() else 0.01
        p = 0.99 if a.priority.lower() == t.expected_priority.lower() else 0.01
        return clamp(0.5 * c + 0.5 * p)
    except Exception:
        return 0.5

def grade_triage(ticket: Any, action: Any) -> float:
    try:
        t = _safe_ticket(ticket)
        a = _safe_action(action)
        c = 0.99 if a.category.lower() == t.expected_category.lower() else 0.01
        p = 0.99 if a.priority.lower() == t.expected_priority.lower() else 0.01
        cl = 0.99 if a.needs_clarification == t.ambiguous else 0.01
        es = 0.99 if a.escalation == t.requires_escalation else 0.01
        return clamp(0.3 * c + 0.3 * p + 0.2 * cl + 0.2 * es)
    except Exception:
        return 0.5

def grade_resolve(ticket: Any, action: Any) -> float:
    try:
        t = _safe_ticket(ticket)
        a = _safe_action(action)
        base = grade_triage(t, a)
        pol = policy_check(t.text, a.response, a.escalation)
        resp = 0.99 if len(a.response) > 50 else 0.01
        return clamp(0.2 * base + 0.4 * pol + 0.4 * resp)
    except Exception:
        return 0.5

def grade_episode(ticket: Any, action: Any, task: str = "triage") -> dict:
    try:
        if task == "classify":
            score = grade_classify(ticket, action)
        elif task == "triage":
            score = grade_triage(ticket, action)
        elif task == "resolve":
            score = grade_resolve(ticket, action)
        else:
            score = grade_triage(ticket, action)
    except Exception:
        score = 0.5

    score = clamp(score)
    return {
        "score": score,
        "total_score": score,
        "task": task
    }
