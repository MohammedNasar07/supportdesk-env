from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
import random
import uvicorn
from graders import grade   # your original grading module

app = FastAPI()

TICKETS = [
    {
      "ticket_id": "TKT-001",
      "subject": "Unable to log in to my account",
      "body": "Every time I try to log in I get a 500 Internal Server Error. This started about an hour ago. I have cleared my cache and tried a different browser. My account is alice@startup.io.",
      "sender_email": "alice@startup.io",
      "timestamp": "2025-03-23T10:15:00Z",
      "gt_category": "account",
      "gt_priority": "high",
      "gt_team": "tech_support",
      "response_keywords": ["login", "500 error", "investigate", "apologize", "email"]
    },
    {
      "ticket_id": "TKT-002",
      "subject": "Reset password link not working",
      "body": "I requested a password reset but the link in the email leads to a 404 page. I have tried 3 times now. Please help me get back into my account.",
      "sender_email": "bob.builder@gmail.com",
      "timestamp": "2025-03-22T14:30:00Z",
      "gt_category": "account",
      "gt_priority": "medium",
      "gt_team": "tech_support",
      "response_keywords": ["password", "reset", "link", "404", "manual reset"]
    },
    {
      "ticket_id": "TKT-003",
      "subject": "Unauthorized access suspicion",
      "body": "I received a notification about a login from a device in Russia. I am currently in London. I think my account has been hacked. Please lock it immediately.",
      "sender_email": "security.first@protonmail.com",
      "timestamp": "2025-03-23T02:00:00Z",
      "gt_category": "account",
      "gt_priority": "critical",
      "gt_team": "tech_support",
      "response_keywords": ["security", "hacked", "lock", "unauthorized", "verify"]
    },
    {
      "ticket_id": "TKT-004",
      "subject": "Error when upgrading to Pro plan",
      "body": "I am trying to upgrade to the Pro plan but the checkout button is greyed out. My credit card on file is valid. Please resolve this so I can access the new features.",
      "sender_email": "pro.user@company.com",
      "timestamp": "2025-03-21T11:45:00Z",
      "gt_category": "billing",
      "gt_priority": "high",
      "gt_team": "billing_team",
      "response_keywords": ["upgrade", "checkout", "billing", "fix", "pro plan"]
    },
    {
      "ticket_id": "TKT-005",
      "subject": "Missing invoice for February",
      "body": "I haven't received my invoice for the month of February. Our accounting department needs it for the monthly close. Can you please send it to me as a PDF?",
      "sender_email": "finance@enterprise.com",
      "timestamp": "2025-03-05T09:00:00Z",
      "gt_category": "billing",
      "gt_priority": "medium",
      "gt_team": "billing_team",
      "response_keywords": ["invoice", "February", "PDF", "billing", "send"]
    },
    {
      "ticket_id": "TKT-025",
      "subject": "Service completely down revenue impact",
      "body": "URGENT. All our users are unable to access the platform. We have been down for 45 minutes. This is causing direct revenue loss estimated at 10000 dollars per hour. Need immediate escalation to engineering.",
      "sender_email": "cto@criticalclient.com",
      "timestamp": "2025-03-23T13:00:00Z",
      "gt_category": "technical",
      "gt_priority": "critical",
      "gt_team": "tech_support",
      "response_keywords": ["outage", "engineering", "escalate", "update", "priority"]
    },
    {
      "ticket_id": "TKT-011",
      "subject": "Absolutely unacceptable service escalation required",
      "body": "This is the THIRD time I am writing about the same issue and nobody has resolved it. My data has been corrupted for 2 weeks, your support keeps closing tickets without fixing anything. I am considering legal action.",
      "sender_email": "angry.client@lawfirm.com",
      "timestamp": "2025-03-22T11:30:00Z",
      "gt_category": "complaint",
      "gt_priority": "critical",
      "gt_team": "management",
      "response_keywords": ["apologize", "escalate", "manager", "priority", "resolve"]
    }
    # ... Simplified for space, but logic applies to full list
]

class EnvState:
    def __init__(self):
        self.ticket = None
        self.actions: List[Dict] = []
        self.step_count = 0
        self.done = False

state = EnvState()

class Action(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def get_tasks():
    from tasks import ClassifyTask, TriageTask, ResolveTask
    return [
        {"id": "classify", "name": "classify", "description": "Classify task", "difficulty": "easy", "grader": "graders:ClassifyGrader"},
        {"id": "triage", "name": "triage", "description": "Triage task", "difficulty": "medium", "grader": "graders:TriageGrader"},
        {"id": "resolve", "name": "resolve", "description": "Resolve task", "difficulty": "hard", "grader": "graders:ResolveGrader"}
    ]

@app.post("/reset")
def reset(task: str = "classify", seed: int = 42):
    random.seed(seed)
    state.ticket = random.choice(TICKETS)
    state.actions = []
    state.step_count = 0
    state.done = False
    return {"observation": state.ticket["body"], "done": False}

@app.post("/step")
def step(action: Action):
    if state.done:
        return {"reward": 0.0, "done": True, "observation": "done"}
    
    # Manual parsing as per original SupportDesk logic
    msg = action.message.lower()
    parsed_action = {"action_type": "unknown"}
    
    if "account" in msg or "billing" in msg or "technical" in msg or "refund" in msg or "complaint" in msg or "general" in msg:
        parsed_action["action_type"] = "classify"
        for cat in ["account", "billing", "technical", "refund", "complaint", "general"]:
            if cat in msg: parsed_action["category"] = cat

    if "critical" in msg or "high" in msg or "medium" in msg or "low" in msg:
        parsed_action["action_type"] = "set_priority"
        for p in ["critical", "high", "medium", "low"]:
            if p in msg: parsed_action["priority"] = p

    if "support" in msg or "finance" in msg or "engineering" in msg:
        parsed_action["action_type"] = "route"
        for t in ["support", "finance", "engineering"]:
            if t in msg: parsed_action["team"] = t

    if len(msg) > 20:
        parsed_action["action_type"] = "draft_response"
        parsed_action["response_draft"] = action.message

    state.actions.append(parsed_action)
    state.step_count += 1
    
    if state.step_count >= 3 or "submit" in msg:
        state.done = True
        scores = grade("resolve", state.actions, state.ticket)
        return {
            "reward": scores["total"],
            "done": True,
            "observation": "Task completed",
            "info": {"final_scores": scores}
        }
    
    return {"reward": 0.1, "done": False, "observation": "Continue", "info": {}}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
