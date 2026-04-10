from .schemas import Ticket, AgentAction
from .policy import policy_check

def clamp(score: float) -> float:
    return max(0.01, min(0.99, score))

def grade_classify(ticket: Ticket, action: AgentAction) -> dict:
    # 50% category, 50% priority
    c = 0.99 if action.category.lower() == ticket.expected_category.lower() else 0.01
    p = 0.99 if action.priority.lower() == ticket.expected_priority.lower() else 0.01
    return {"total_score": clamp(0.5 * c + 0.5 * p)}

def grade_triage(ticket: Ticket, action: AgentAction) -> dict:
    # 30% category, 30% priority, 40% clarification/escalation
    c = 0.99 if action.category.lower() == ticket.expected_category.lower() else 0.01
    p = 0.99 if action.priority.lower() == ticket.expected_priority.lower() else 0.01
    cl = 0.99 if action.needs_clarification == ticket.ambiguous else 0.01
    es = 0.99 if action.escalation == ticket.requires_escalation else 0.01
    return {"total_score": clamp(0.3 * c + 0.3 * p + 0.2 * cl + 0.2 * es)}

def grade_resolve(ticket: Ticket, action: AgentAction) -> dict:
    # 20% triage, 40% policy, 40% response quality
    base = grade_triage(ticket, action)["total_score"]
    pol = policy_check(ticket.text, action.response, action.escalation)
    resp = 0.99 if len(action.response) > 50 else 0.01
    return {"total_score": clamp(0.2 * base + 0.4 * pol + 0.4 * resp)}

def grade_episode(ticket: Ticket, action: AgentAction, task: str = "triage") -> dict:
    if task == "classify":
        return grade_classify(ticket, action)
    if task == "triage":
        return grade_triage(ticket, action)
    if task == "resolve":
        return grade_resolve(ticket, action)
    return grade_triage(ticket, action)
