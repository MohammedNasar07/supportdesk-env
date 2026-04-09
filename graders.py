from __future__ import annotations
from typing import Any, Dict, List, Optional

# Phase 2 safety limits
MIN_VAL = 0.0001
MAX_VAL = 0.9999


def clamp(x: float) -> float:
    try:
        v = float(x)
    except:
        return MIN_VAL
    return min(max(v, MIN_VAL), MAX_VAL)


_PRIORITY_ORDER = ["low", "medium", "high", "critical"]


def _category_score(predicted: Optional[str], ground_truth: str) -> float:
    if not predicted:
        return 0.1
    return 0.9 if predicted.lower() == ground_truth.lower() else 0.3


def _priority_score(predicted: Optional[str], ground_truth: str) -> float:
    if not predicted:
        return 0.1
    p, gt = (predicted or "").lower(), ground_truth.lower()
    if p not in _PRIORITY_ORDER:
        return 0.2
    diff = abs(_PRIORITY_ORDER.index(p) - _PRIORITY_ORDER.index(gt))
    return {0: 0.9, 1: 0.6}.get(diff, 0.3)


def _team_score(predicted: Optional[str], ground_truth: str) -> float:
    if not predicted:
        return 0.1
    return 0.9 if (predicted or "").lower() == ground_truth.lower() else 0.3


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
    if any(word in text for word in ["thank", "hello"]):
        score += 0.2
    return score


def grade_classify(actions_taken, ticket):
    category = next(
        (
            a.get("category")
            for a in actions_taken
            if a.get("action_type") == "classify"
        ),
        None,
    )
    return {"total": clamp(_category_score(category, ticket["gt_category"]))}


def grade_triage(actions_taken, ticket):
    cat = _category_score(
        next(
            (
                a.get("category")
                for a in actions_taken
                if a.get("action_type") == "classify"
            ),
            None,
        ),
        ticket["gt_category"],
    )
    pri = _priority_score(
        next(
            (
                a.get("priority")
                for a in actions_taken
                if a.get("action_type") == "set_priority"
            ),
            None,
        ),
        ticket["gt_priority"],
    )
    tm = _team_score(
        next(
            (a.get("team") for a in actions_taken if a.get("action_type") == "route"),
            None,
        ),
        ticket["gt_team"],
    )
    return {"total": clamp((0.4 * cat) + (0.3 * pri) + (0.3 * tm))}


def grade_resolve(actions_taken, ticket):
    cat = _category_score(
        next(
            (
                a.get("category")
                for a in actions_taken
                if a.get("action_type") == "classify"
            ),
            None,
        ),
        ticket["gt_category"],
    )
    pri = _priority_score(
        next(
            (
                a.get("priority")
                for a in actions_taken
                if a.get("action_type") == "set_priority"
            ),
            None,
        ),
        ticket["gt_priority"],
    )
    tm = _team_score(
        next(
            (a.get("team") for a in actions_taken if a.get("action_type") == "route"),
            None,
        ),
        ticket["gt_team"],
    )
    resp = _response_quality_score(
        next(
            (
                a.get("response_draft")
                for a in actions_taken
                if a.get("action_type") == "draft_response"
            ),
            None,
        ),
        ticket.get("response_keywords", []),
    )
    return {"total": clamp((0.2 * cat) + (0.15 * pri) + (0.15 * tm) + (0.5 * resp))}


def grade(task_name, actions_taken, ticket):
    graders = {
        "classify": grade_classify,
        "triage": grade_triage,
        "resolve": grade_resolve,
    }
    if task_name not in graders:
        raise ValueError(task_name)
    return graders[task_name](actions_taken, ticket)
