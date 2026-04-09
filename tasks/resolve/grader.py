"""Grader for the resolve task."""

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


def _response_score(resp, keywords):
    if not resp:
        return 0.2
    text = resp.lower()
    score = 0.3
    if len(text) > 50:
        score += 0.2
    if keywords:
        hits = sum(1 for k in keywords if k.lower() in text)
        score += 0.3 * (hits / len(keywords))
    if "thank" in text or "hello" in text:
        score += 0.2
    return _safe(score)


def grade(actions, ticket):
    """Grade the resolve task. Returns dict with 'total' score."""
    cat = pri = team = resp = None

    for a in actions:
        t = a.get("action_type")
        if t == "classify":
            cat = a.get("category")
        elif t == "set_priority":
            pri = a.get("priority")
        elif t == "route":
            team = a.get("team")
        elif t == "draft_response":
            resp = a.get("response_draft")

    total = (
        0.2 * _category_score(cat, ticket["gt_category"])
        + 0.15 * _priority_score(pri, ticket["gt_priority"])
        + 0.15 * _team_score(team, ticket["gt_team"])
        + 0.5 * _response_score(resp, ticket.get("response_keywords", []))
    )

    return {"total": _safe(total)}
