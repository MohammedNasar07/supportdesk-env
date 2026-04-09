# tasks.py
# Task definitions — plain Python, no local imports.

from __future__ import annotations
from typing import Any, Dict

TASKS: Dict[str, Dict[str, Any]] = {
    "classify": {
        "id":          "classify",
        "name":        "classify",
        "description": (
            "Read the support ticket and classify it into exactly one category: "
            "billing, technical, general, refund, or complaint. "
            "Take a classify action with the correct category, then submit."
        ),
        "difficulty":       "easy",
        "required_actions": ["classify", "submit"],
        "max_steps":        4,
        "grader":           "tasks.classify.grader:grade",
        "hint": (
            "billing = charges/invoices/plans/pricing. "
            "technical = bugs/errors/API/login/security/integrations. "
            "general = how-to/feature-requests/cancellations/data-policies. "
            "refund = explicit money-back requests. "
            "complaint = dissatisfaction/rudeness/legal threats/escalation."
        ),
    },
    "triage": {
        "id":          "triage",
        "name":        "triage",
        "description": (
            "Perform full triage: (1) classify category, "
            "(2) set priority (critical/high/medium/low), "
            "(3) route to the correct team, then (4) submit. "
            "All three actions required for full score."
        ),
        "difficulty":       "medium",
        "required_actions": ["classify", "set_priority", "route", "submit"],
        "max_steps":        6,
        "grader":           "tasks.triage.grader:grade",
        "hint": (
            "Priority: critical=production down/security breach/legal threats, "
            "high=billing errors/account locked/urgent deadlines, "
            "medium=partial issues/billing questions/moderate complaints, "
            "low=how-to/feature requests. "
            "Teams: billing_team=charges/refunds/invoices, "
            "tech_support=bugs/API/login/security/integrations, "
            "customer_success=cancellations/enterprise-sales, "
            "management=complaints/escalations, "
            "general_support=how-to/policies/feature-requests."
        ),
    },
    "resolve": {
        "id":          "resolve",
        "name":        "resolve",
        "description": (
            "Fully resolve the ticket: (1) classify, (2) set priority, "
            "(3) route to team, (4) draft a professional customer-facing response "
            "between 60 and 800 characters, then (5) submit. "
            "Response must acknowledge the issue, show empathy, describe next steps."
        ),
        "difficulty":       "hard",
        "required_actions": ["classify", "set_priority", "route", "draft_response", "submit"],
        "max_steps":        8,
        "grader":           "tasks.resolve.grader:grade",
        "hint": (
            "Good responses: greet the customer, acknowledge the specific issue, "
            "apologize if appropriate, state concrete next steps, professional sign-off. "
            "Be specific to the ticket. Avoid placeholders like [Your Name]."
        ),
    },
}


def get_task(name: str) -> Dict[str, Any]:
    if name not in TASKS:
        raise ValueError(
            f"Unknown task: {name!r}. Valid tasks: {list(TASKS.keys())}"
        )
    return TASKS[name]
