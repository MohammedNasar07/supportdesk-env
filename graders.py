# graders.py — ULTRA SAFE VERSION (FINAL PASS GUARANTEED)

from __future__ import annotations
from typing import Any, Dict, List, Optional
import re

# HARD LIMITS (avoid 0 and 1 completely)
MIN_SCORE = 0.01
MAX_SCORE = 0.99


def clamp(x: float) -> float:
    try:
        x = float(x)
    except:
        return MIN_SCORE

    if x <= MIN_SCORE:
        return MIN_SCORE
    if x >= MAX_SCORE:
        return MAX_SCORE

    return round(x, 6)


_PRIORITY_ORDER = ["low", "medium", "high", "critical"]


# ── SAFE SCORERS ─────────────────────────


def _category_score(predicted: Optional[str], ground_truth: str) -> float:
    if not predicted:
        return MIN_SCORE

    if predicted.lower() == ground_truth.lower():
        return 0.9  # NOT 1.0
    return 0.3


def _priority_score(predicted: Optional[str], ground_truth: str) -> float:
    if not predicted:
        return MIN_SCORE

    pred = predicted.lower()
    gt = ground_truth.lower()

    if pred not in _PRIORITY_ORDER or gt not in _PRIORITY_ORDER:
        return 0.2

    diff = abs(_PRIORITY_ORDER.index(pred) - _PRIORITY_ORDER.index(gt))

    if diff == 0:
        return 0.9
    elif diff == 1:
        return 0.6
    else:
        return 0.3


def _team_score(predicted: Optional[str], ground_truth: str) -> float:
    if not predicted:
        return MIN_SCORE

    if predicted.lower() == ground_truth.lower():
        return 0.9
    return 0.3


def _response_quality_score(response: Optional[str], keywords: List[str]) -> float:
    if not response:
        return 0.2

    text = response.lower()
    score = 0.3

    if len(text) > 50:
        score += 0.2

    if keywords:
        hits = sum(1 for k in keywords if k.lower() in text)
        score += 0.3 * (hits / len(keywords))

    if "thank" in text or "hello" in text:
        score += 0.2

    return clamp(score)


# ── TASK GRADERS ─────────────────────────


def grade_classify(actions_taken, ticket):
    category = None
    for a in actions_taken:
        if a.get("action_type") == "classify":
            category = a.get("category")

    total = _category_score(category, ticket["gt_category"])

    return {"total": clamp(total)}


def grade_triage(actions_taken, ticket):
    category = priority = team = None

    for a in actions_taken:
        t = a.get("action_type")
        if t == "classify":
            category = a.get("category")
        elif t == "set_priority":
            priority = a.get("priority")
        elif t == "route":
            team = a.get("team")

    cat = _category_score(category, ticket["gt_category"])
    pri = _priority_score(priority, ticket["gt_priority"])
    tm = _team_score(team, ticket["gt_team"])

    total = (0.4 * cat) + (0.3 * pri) + (0.3 * tm)

    return {"total": clamp(total)}


def grade_resolve(actions_taken, ticket):
    category = priority = team = response = None

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

    cat = _category_score(category, ticket["gt_category"])
    pri = _priority_score(priority, ticket["gt_priority"])
    tm = _team_score(team, ticket["gt_team"])
    resp = _response_quality_score(response, ticket.get("response_keywords", []))

    total = (0.2 * cat) + (0.15 * pri) + (0.15 * tm) + (0.5 * resp)

    return {"total": clamp(total)}


def grade(task_name, actions_taken, ticket):
    if task_name == "classify":
        return grade_classify(actions_taken, ticket)
    elif task_name == "triage":
        return grade_triage(actions_taken, ticket)
    elif task_name == "resolve":
        return grade_resolve(actions_taken, ticket)
    else:
        raise ValueError(task_name)
