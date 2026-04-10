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

def _response_score(resp, keywords):
    if not resp: return 0.2
    text = resp.lower()
    score = 0.3
    if len(text) > 50: score += 0.2
    if keywords:
        hits = sum(1 for k in keywords if k.lower() in text)
        score += 0.3 * (hits / len(keywords))
    if "thank" in text or "hello" in text: score += 0.2
    return min(1.0, score)

def grade(actions, ticket):
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

    score = (
        0.2 * _category_score(cat, ticket["gt_category"])
        + 0.15 * _priority_score(pri, ticket["gt_priority"])
        + 0.15 * _team_score(team, ticket["gt_team"])
        + 0.5 * _response_score(resp, ticket.get("response_keywords", []))
    )
    return {"total": _safe(score)}
