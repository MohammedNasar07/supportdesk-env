# Grader Logic (formerly graders.py)
def _safe(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.01
    return max(0.01, min(0.99, round(x, 6)))

def _category_score(predicted, gt):
    if not predicted: return 0.2
    return 1.0 if str(predicted).lower() == str(gt).lower() else 0.3

def _priority_score(predicted, gt):
    order = ["low", "medium", "high", "critical"]
    if not predicted: return 0.2
    p = str(predicted).lower()
    g = str(gt).lower()
    if p not in order or g not in order: return 0.3
    diff = abs(order.index(p) - order.index(g))
    if diff == 0: return 1.0
    if diff == 1: return 0.5
    return 0.3

def _team_score(predicted, gt):
    if not predicted: return 0.2
    return 1.0 if str(predicted).lower() == str(gt).lower() else 0.3

def _response_score(resp, keywords):
    if not resp: return 0.2
    text = str(resp).lower()
    score = 0.3
    if len(text) > 50: score += 0.2
    if keywords:
        hits = sum(1 for k in keywords if str(k).lower() in text)
        score += 0.3 * (hits / len(keywords))
    if "thank" in text or "hello" in text: score += 0.2
    return min(1.0, score)

def grade(task_type, actions, ticket):
    cat = pri = team = resp = None
    for a in actions:
        t = a.get("action_type") or a.get("category") or a.get("priority")
        if "category" in a or t == "classify":
            cat = a.get("category")
        if "priority" in a or t == "set_priority":
            pri = a.get("priority")
        if "team" in a or t == "route":
            team = a.get("team")
        if "response_draft" in a or t == "draft_response":
            resp = a.get("response_draft")

    score = (
        0.2 * _category_score(cat, ticket.get("gt_category", ""))
        + 0.15 * _priority_score(pri, ticket.get("gt_priority", ""))
        + 0.15 * _team_score(team, ticket.get("gt_team", ""))
        + 0.5 * _response_score(resp, ticket.get("response_keywords", []))
    )
    return {"total": _safe(score)}

# OpenEnv Registrations
try:
    from openenv import Task
    from .classify_grader import ClassifyGrader
    from .triage_grader import TriageGrader
    from .resolve_grader import ResolveGrader

    Task.register_grader("classify", ClassifyGrader)
    Task.register_grader("triage", TriageGrader)
    Task.register_grader("resolve", ResolveGrader)
except ImportError:
    pass
