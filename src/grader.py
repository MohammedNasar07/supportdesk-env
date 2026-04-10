from .schemas import Ticket, AgentAction
from .policy import policy_check

def clamp(score: float) -> float:
    """
    Mandatory OpenEnv validator constraint: Clamp scores between 0.01 and 0.99.
    """
    return max(0.01, min(0.99, score))

def score_match(pred: str, gold: str) -> float:
    return 0.99 if pred.strip().lower() == gold.strip().lower() else 0.01

def score_bool(pred: bool, gold: bool) -> float:
    return 0.99 if pred == gold else 0.01

def grade_episode(ticket: Ticket, action: AgentAction) -> dict:
    """
    Weighted grading logic:
    30% Category
    20% Priority
    20% Clarification
    15% Escalation
    15% Policy Safety
    """
    c_score = score_match(action.category, ticket.expected_category)
    p_score = score_match(action.priority, ticket.expected_priority)
    cl_score = score_bool(action.needs_clarification, ticket.ambiguous)
    e_score = score_bool(action.escalation, ticket.requires_escalation)
    pol_score = policy_check(ticket.text, action.response, action.escalation)
    
    # Weighted calculation
    weighted_total = (
        0.30 * c_score +
        0.20 * p_score +
        0.20 * cl_score +
        0.15 * e_score +
        0.15 * pol_score
    )
    
    final_score = clamp(weighted_total)
    
    return {
        "category_score": c_score,
        "priority_score": p_score,
        "clarification_score": cl_score,
        "escalation_score": e_score,
        "policy_score": pol_score,
        "total_score": final_score
    }
