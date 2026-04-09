# graders.py — FINAL VALIDATOR SAFE VERSION

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

EPS = 1e-6

_PRIORITY_ORDER = ["low", "medium", "high", "critical"]


# ───────────────── SAFE SCORE ─────────────────


def safe_score(x: float) -> float:
    try:
        x = float(x)
    except:
        x = 0.0

    if x <= 0.0:
        return EPS
    if x >= 1.0:
        return 1 - EPS
    return x


# ───────────────── HELPERS ─────────────────


def _category_score(predicted: Optional[str], ground_truth: str) -> float:
    if not predicted:
        return 0.0
    return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0


def _priority_score(predicted: Optional[str], ground_truth: str) -> float:
    if not predicted:
        return 0.0

    pred = predicted.lower()
    gt = ground_truth.lower()

    if pred not in _PRIORITY_ORDER or gt not in _PRIORITY_ORDER:
        return 0.0

    diff = abs(_PRIORITY_ORDER.index(pred) - _PRIORITY_ORDER.index(gt))

    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.5
    else:
        return 0.0


def _team_score(predicted: Optional[str], ground_truth: str) -> float:
    if not predicted:
        return 0.0
    return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0


def _response_quality_score(response: Optional[str], keywords: List[str]) -> float:
    if not response:
        return 0.0

    text = response.lower()
    score = 0.0

    # length
    if 60 <= len(text) <= 800:
        score += 0.25

    # keyword coverage
    if keywords:
        hits = sum(1 for k in keywords if k.lower() in text)
        score += 0.40 * (hits / len(keywords))

    # professionalism
    if any(x in text for x in ["hello", "hi", "dear"]):
        score += 0.1
    if any(x in text for x in ["regards", "thank", "help"]):
        score += 0.1

    # no placeholders
    if "todo" not in text and "[" not in text:
        score += 0.15

    return min(score, 1.0)


# ───────────────── TASKS ─────────────────


def grade_classify(actions_taken, ticket):
    category = None
    for a in actions_taken:
        if a.get("action_type") == "classify":
            category = a.get("category")

    cat = _category_score(category, ticket["gt_category"])

    total = safe_score(cat)

    return {"total": total}


def grade_triage(actions_taken, ticket):
    category = priority = team = None

    for a in actions_taken:
        if a.get("action_type") == "classify":
            category = a.get("category")
        elif a.get("action_type") == "set_priority":
            priority = a.get("priority")
        elif a.get("action_type") == "route":
            team = a.get("team")

    cat = _category_score(category, ticket["gt_category"])
    pr = _priority_score(priority, ticket["gt_priority"])
    tm = _team_score(team, ticket["gt_team"])

    raw = 0.4 * cat + 0.3 * pr + 0.3 * tm

    total = safe_score(raw)

    return {"total": total}


def grade_resolve(actions_taken, ticket):
    category = priority = team = response = None

    for a in actions_taken:
        if a.get("action_type") == "classify":
            category = a.get("category")
        elif a.get("action_type") == "set_priority":
            priority = a.get("priority")
        elif a.get("action_type") == "route":
            team = a.get("team")
        elif a.get("action_type") == "draft_response":
            response = a.get("response_draft")

    cat = _category_score(category, ticket["gt_category"])
    pr = _priority_score(priority, ticket["gt_priority"])
    tm = _team_score(team, ticket["gt_team"])
    resp = _response_quality_score(response, ticket.get("response_keywords", []))

    raw = 0.2 * cat + 0.15 * pr + 0.15 * tm + 0.5 * resp

    total = safe_score(raw)

    return {"total": total}


# ───────────────── DISPATCH ─────────────────


def grade(task_name, actions_taken, ticket):
    if task_name == "classify":
        return grade_classify(actions_taken, ticket)
    elif task_name == "triage":
        return grade_triage(actions_taken, ticket)
    elif task_name == "resolve":
        return grade_resolve(actions_taken, ticket)
    else:
        raise ValueError(f"Unknown task: {task_name}")
