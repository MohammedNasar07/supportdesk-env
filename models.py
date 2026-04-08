# models.py
# All Pydantic models for SupportDesk-Env.
# No relative imports. No package. Works when run from any folder.

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class Category(str, Enum):
    BILLING   = "billing"
    TECHNICAL = "technical"
    GENERAL   = "general"
    REFUND    = "refund"
    COMPLAINT = "complaint"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"


class Team(str, Enum):
    BILLING_TEAM     = "billing_team"
    TECH_SUPPORT     = "tech_support"
    CUSTOMER_SUCCESS = "customer_success"
    MANAGEMENT       = "management"
    GENERAL_SUPPORT  = "general_support"


class ActionType(str, Enum):
    CLASSIFY       = "classify"
    SET_PRIORITY   = "set_priority"
    ROUTE          = "route"
    DRAFT_RESPONSE = "draft_response"
    SUBMIT         = "submit"


# ── Action ────────────────────────────────────────────────────────────────────

class TriageAction(BaseModel):
    """
    One action per step.

    action_type      | required extra field
    -----------------|---------------------
    classify         | category
    set_priority     | priority
    route            | team
    draft_response   | response_draft
    submit           | (none — ends episode)
    """
    action_type:    ActionType
    category:       Optional[Category] = None
    priority:       Optional[Priority] = None
    team:           Optional[Team]     = None
    response_draft: Optional[str]      = None


# ── Observation ───────────────────────────────────────────────────────────────

class TicketObservation(BaseModel):
    """Everything the agent sees each step."""
    ticket_id:        str
    subject:          str
    body:             str
    sender_email:     str
    timestamp:        str
    task_name:        str
    task_description: str
    current_step:     int
    max_steps:        int
    actions_taken:    List[Dict[str, Any]] = Field(default_factory=list)
    current_score:    float                = 0.0
    required_actions: List[str]            = Field(default_factory=list)
    hint:             Optional[str]        = None


# ── State ─────────────────────────────────────────────────────────────────────

class EnvState(BaseModel):
    task_name:     str
    ticket_id:     str
    step:          int
    done:          bool
    total_reward:  float
    actions_taken: List[Dict[str, Any]]
    solution:      Optional[Dict[str, Any]] = None


# ── Step result ───────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: TicketObservation
    reward:      float
    done:        bool
    info:        Dict[str, Any] = Field(default_factory=dict)
