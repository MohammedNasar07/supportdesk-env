def _safe(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, round(x, 6)))

def grade(actions, ticket):
    """Grade the classify task. Returns dict with 'total' score."""
    cat = None
    for a in actions:
        if a.get("action_type") == "classify":
            cat = a.get("category")

    if not cat:
        score = 0.2
    elif cat.lower() == ticket["gt_category"].lower():
        score = 1.0  # Full credit for correct category
    else:
        score = 0.3

    return {"total": _safe(score)}
