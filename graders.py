# graders.py — FINAL FIXED VERSION (NO 0 / NO 1)

from typing import Any, Dict, List, Optional
import re


# NEVER allow 0 or 1
def safe(x: float) -> float:
    try:
        x = float(x)
    except:
        return 0.01

    if x <= 0.0:
        return 0.01
    if x >= 1.0:
        return 0.99

    return round(x, 6)


_PRIORITY_ORDER = ["low", "medium", "high", "critical"]


# ───────── SCORING ─────────


def _category_score(predicted: Optional[str], gt: str) -> float:
    if not predicted:
        return 0.2
    return 0.95 if predicted.lower() == gt.lower() else 0.3


def _priority_score(predicted: Optional[str], gt: str) -> float:
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


def _team_score(predicted: Optional[str], gt: str) -> float:
    if not predicted:
        return 0.2
    return 0.95 if predicted.lower() == gt.lower() else 0.3


def _response_score(resp: Optional[str], keywords: List[str]) -> float:
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

    return safe(score)


# ───────── TASKS ─────────


def grade_classify(actions, ticket):
    cat = None
    for a in actions:
        if a.get("action_type") == "classify":
            cat = a.get("category")

    total = _category_score(cat, ticket["gt_category"])
    return {"total": safe(total)}


def grade_triage(actions, ticket):
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

    return {"total": safe(total)}


def grade_resolve(actions, ticket):
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

    return {"total": safe(total)}


def grade(task, actions, ticket):
    if task == "classify":
        return grade_classify(actions, ticket)
    elif task == "triage":
        return grade_triage(actions, ticket)
    elif task == "resolve":
        return grade_resolve(actions, ticket)
    else:
        raise ValueError(task)
