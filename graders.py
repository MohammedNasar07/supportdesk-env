from __future__ import annotations
from typing import Any, Dict, List, Optional

MIN_SCORE = 0.01
MAX_SCORE = 0.99


def clamp(x: float) -> float:
    try:
        val = float(x)
    except Exception:
        return MIN_SCORE

    if val <= MIN_SCORE:
        return MIN_SCORE
    if val >= MAX_SCORE:
        return MAX_SCORE

    return round(val, 4)


_PRIORITY_ORDER = ["low", "medium", "high", "critical"]

# ── SAFE SCORERS ─────────────────────────


def _category_score(predicted: Optional[str], ground_truth: str) -> float:
    if not predicted:
        return MIN_SCORE
    gt_val = ground_truth if ground_truth else ""
    if predicted.lower() == gt_val.lower():
        return 0.85
    return 0.15


def _priority_score(predicted: Optional[str], ground_truth: str) -> float:
    if not predicted:
        return MIN_SCORE
    pred = predicted.lower()
    gt_val = ground_truth.lower() if ground_truth else ""

    if pred not in _PRIORITY_ORDER or gt_val not in _PRIORITY_ORDER:
        return 0.15

    diff = abs(_PRIORITY_ORDER.index(pred) - _PRIORITY_ORDER.index(gt_val))

    if diff == 0:
        return 0.85
    elif diff == 1:
        return 0.55
    else:
        return 0.25


def _team_score(predicted: Optional[str], ground_truth: str) -> float:
    if not predicted:
        return MIN_SCORE
    gt_val = ground_truth if ground_truth else ""
    if predicted.lower() == gt_val.lower():
        return 0.85
    return 0.15


def _response_quality_score(response: Optional[str], keywords: List[str]) -> float:
    if not response:
        return MIN_SCORE

    text = response.lower()
    score = 0.25

    if len(text) > 50:
        score += 0.25

    if keywords:
        hits = sum([2 for k in keywords if k.lower() in text])
        ratio = hits / (len(keywords) * 2)
        score += 0.25 * ratio

    if "thank" in text or "hello" in text:
        score += 0.15

    return clamp(score)


# ── TASK GRADERS ─────────────────────────


def grade_classify(
    actions_taken: List[Dict[str, Any]], ticket: Dict[str, Any]
) -> Dict[str, float]:
    category = None
    for a in actions_taken:
        if a.get("action_type") == "classify":
            category = a.get("category")

    total = _category_score(category, ticket.get("gt_category", ""))
    return {"total": clamp(total)}


def grade_triage(
    actions_taken: List[Dict[str, Any]], ticket: Dict[str, Any]
) -> Dict[str, float]:
    category = None
    priority = None
    team = None

    for a in actions_taken:
        t = a.get("action_type")
        if t == "classify":
            category = a.get("category")
        elif t == "set_priority":
            priority = a.get("priority")
        elif t == "route":
            team = a.get("team")

    cat = _category_score(category, ticket.get("gt_category", ""))
    pri = _priority_score(priority, ticket.get("gt_priority", ""))
    tm = _team_score(team, ticket.get("gt_team", ""))

    total = (0.4 * cat) + (0.3 * pri) + (0.3 * tm)
    return {"total": clamp(total)}


def grade_resolve(
    actions_taken: List[Dict[str, Any]], ticket: Dict[str, Any]
) -> Dict[str, float]:
    category = None
    priority = None
    team = None
    response = None

    for a in actions_taken:
        t = a.get("action_type")
        if t == "classify":
            category = a.get("category")
        elif t == "set_priority":
            priority = a.get("priority")
        elif t == "route":
            team = a.get("team")
        elif t == "draft_response":
            response = a.get("response_draft")

    cat = _category_score(category, ticket.get("gt_category", ""))
    pri = _priority_score(priority, ticket.get("gt_priority", ""))
    tm = _team_score(team, ticket.get("gt_team", ""))
    resp = _response_quality_score(response, ticket.get("response_keywords", []))

    total = (0.2 * cat) + (0.15 * pri) + (0.15 * tm) + (0.5 * resp)
    return {"total": clamp(total)}


def grade(
    task_name: str, actions_taken: List[Dict[str, Any]], ticket: Dict[str, Any]
) -> Dict[str, float]:
    if task_name == "classify":
        return grade_classify(actions_taken, ticket)
    elif task_name == "triage":
        return grade_triage(actions_taken, ticket)
    elif task_name == "resolve":
        return grade_resolve(actions_taken, ticket)
    else:
        return {"total": MIN_SCORE}
