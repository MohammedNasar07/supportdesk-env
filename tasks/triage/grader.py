"""Grader for the triage task."""

_PRIORITY_ORDER = ["low", "medium", "high", "critical"]


def _safe(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.01
    if x <= 0.0:
        return 0.01
    if x >= 1.0:
        return 0.99
    return round(x, 6)


def _category_score(predicted, gt):
    if not predicted:
        return 0.2
    return 0.95 if predicted.lower() == gt.lower() else 0.3


def _priority_score(predicted, gt):
    if not predicted:
        return 0.2
    if predicted not in _PRIORITY_ORDER or gt not in _PRIORITY_ORDER:
        return 0.3
    diff = abs(_PRIORITY_ORDER.index(predicted) - _PRIORITY_ORDER.index(gt))
    if diff == 0:
        return 0.95
    elif diff == 1:
        return 0.5
    else:
        return 0.3


def _team_score(predicted, gt):
    if not predicted:
        return 0.2
    return 0.95 if predicted.lower() == gt.lower() else 0.3


def grade(actions, ticket):
    """Grade the triage task. Returns dict with 'total' score."""
    cat = pri = team = None

    for a in actions:
        if a.get("action_type") == "classify":
            cat = a.get("category")
        elif a.get("action_type") == "set_priority":
            pri = a.get("priority")
        elif a.get("action_type") == "route":
            team = a.get("team")

    total = (
        0.4 * _category_score(cat, ticket["gt_category"])
        + 0.3 * _priority_score(pri, ticket["gt_priority"])
        + 0.3 * _team_score(team, ticket["gt_team"])
    )

    return {"total": _safe(total)}
