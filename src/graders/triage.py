def _safe(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, round(x, 6)))

def _category_score(predicted, gt):
    if not predicted: return 0.2
    return 1.0 if predicted.lower() == gt.lower() else 0.3

def _priority_score(predicted, gt):
    order = ["low", "medium", "high", "critical"]
    if not predicted: return 0.2
    if predicted not in order or gt not in order: return 0.3
    diff = abs(order.index(predicted) - order.index(gt))
    if diff == 0: return 1.0
    if diff == 1: return 0.5
    return 0.3

def _team_score(predicted, gt):
    if not predicted: return 0.2
    return 1.0 if predicted.lower() == gt.lower() else 0.3

def grade(actions, ticket):
    cat = pri = team = None
    for a in actions:
        if a.get("action_type") == "classify":
            cat = a.get("category")
        elif a.get("action_type") == "set_priority":
            pri = a.get("priority")
        elif a.get("action_type") == "route":
            team = a.get("team")

    score = (
        0.4 * _category_score(cat, ticket["gt_category"])
        + 0.3 * _priority_score(pri, ticket["gt_priority"])
        + 0.3 * _team_score(team, ticket["gt_team"])
    )
    return {"total": _safe(score)}
