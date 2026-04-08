# graders.py
# Deterministic graders for all three tasks.
# No imports from other local files — fully self-contained.
# All scores are in [0.0, 1.0].

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Priority order from lowest to highest urgency
_PRIORITY_ORDER = ["low", "medium", "high", "critical"]


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _category_score(predicted: Optional[str], ground_truth: str) -> float:
    """1.0 if exact match (case-insensitive), else 0.0."""
    if not predicted:
        return 0.0
    return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0


def _priority_score(predicted: Optional[str], ground_truth: str) -> float:
    """
    1.0 = exact match
    0.5 = one level off  (e.g. predicted high, actual critical)
    0.0 = two or more levels off
    """
    if not predicted:
        return 0.0
    pred = predicted.strip().lower()
    gt   = ground_truth.strip().lower()
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
    """1.0 if exact match (case-insensitive), else 0.0."""
    if not predicted:
        return 0.0
    return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0


def _response_quality_score(
    response: Optional[str],
    keywords: List[str],
) -> float:
    """
    Scores a customer-facing response draft on four axes:
      1. Length adequacy  (0.25) — between 60 and 800 characters
      2. Keyword coverage (0.40) — expected domain keywords present
      3. Professionalism  (0.20) — greeting and sign-off detected
      4. No placeholders  (0.15) — no [Your Name], TODO, etc.
    """
    if not response or not response.strip():
        return 0.0

    text  = response.strip()
    lower = text.lower()
    score = 0.0

    # 1. Length
    length = len(text)
    if 60 <= length <= 800:
        score += 0.25
    elif length > 20:
        score += 0.08

    # 2. Keyword coverage
    if keywords:
        hits = sum(1 for kw in keywords if kw.lower() in lower)
        score += 0.40 * (hits / len(keywords))

    # 3. Professionalism markers
    greetings = [r"\bhello\b", r"\bhi\b", r"\bdear\b", r"\bgreetings\b",
                 r"thank you for", r"thanks for"]
    signoffs  = [r"\bregards\b", r"\bsincerely\b", r"\bbest\b",
                 r"let me know", r"please don.?t hesitate", r"feel free to",
                 r"we.?re here to help"]
    prof = 0.0
    if any(re.search(p, lower) for p in greetings):
        prof += 0.5
    if any(re.search(p, lower) for p in signoffs):
        prof += 0.5
    score += 0.20 * prof

    # 4. No placeholder text
    placeholders = [r"\[your name\]", r"\[agent\]", r"\bTODO\b",
                    r"\bPLACEHOLDER\b", r"\[insert", r"<n>"]
    if not any(re.search(p, text, re.IGNORECASE) for p in placeholders):
        score += 0.15

    return round(min(score, 1.0), 4)


# ── Task graders ──────────────────────────────────────────────────────────────

def grade_classify(actions_taken: List[Dict[str, Any]], ticket: Dict[str, Any]) -> Dict[str, float]:
    """Task 1 — easy. Only category matters."""
    category = None
    for a in actions_taken:
        if a.get("action_type") == "classify":
            category = a.get("category")
    cat_s = _category_score(category, ticket["gt_category"])
    return {"total": round(cat_s, 4), "category": round(cat_s, 4)}


def grade_triage(actions_taken: List[Dict[str, Any]], ticket: Dict[str, Any]) -> Dict[str, float]:
    """Task 2 — medium. Category 40% + Priority 30% + Team 30%."""
    category = priority = team = None
    for a in actions_taken:
        atype = a.get("action_type")
        if atype == "classify":
            category = a.get("category")
        elif atype == "set_priority":
            priority = a.get("priority")
        elif atype == "route":
            team = a.get("team")

    cat_s  = _category_score(category, ticket["gt_category"])
    prio_s = _priority_score(priority, ticket["gt_priority"])
    team_s = _team_score(team,     ticket["gt_team"])
    total  = round(0.40 * cat_s + 0.30 * prio_s + 0.30 * team_s, 4)

    return {
        "total":    total,
        "category": round(cat_s, 4),
        "priority": round(prio_s, 4),
        "team":     round(team_s, 4),
    }


def grade_resolve(actions_taken: List[Dict[str, Any]], ticket: Dict[str, Any]) -> Dict[str, float]:
    """Task 3 — hard. Category 20% + Priority 15% + Team 15% + Response 50%."""
    category = priority = team = response = None
    for a in actions_taken:
        atype = a.get("action_type")
        if atype == "classify":
            category = a.get("category")
        elif atype == "set_priority":
            priority = a.get("priority")
        elif atype == "route":
            team = a.get("team")
        elif atype == "draft_response":
            response = a.get("response_draft")

    cat_s  = _category_score(category, ticket["gt_category"])
    prio_s = _priority_score(priority, ticket["gt_priority"])
    team_s = _team_score(team,     ticket["gt_team"])
    resp_s = _response_quality_score(response, ticket.get("response_keywords", []))

    total = round(
        0.20 * cat_s +
        0.15 * prio_s +
        0.15 * team_s +
        0.50 * resp_s,
        4,
    )
    return {
        "total":            total,
        "category":         round(cat_s, 4),
        "priority":         round(prio_s, 4),
        "team":             round(team_s, 4),
        "response_quality": round(resp_s, 4),
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

def grade(task_name: str, actions_taken: List[Dict[str, Any]], ticket: Dict[str, Any]) -> Dict[str, float]:
    if task_name == "classify":
        return grade_classify(actions_taken, ticket)
    elif task_name == "triage":
        return grade_triage(actions_taken, ticket)
    elif task_name == "resolve":
        return grade_resolve(actions_taken, ticket)
    else:
        raise ValueError(f"Unknown task: {task_name!r}")
