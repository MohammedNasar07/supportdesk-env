# environment.py
# SupportDeskEnv — the main environment class.
# All imports are absolute (direct module names). No relative imports.

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Direct imports — all files are in the SAME folder
import ticket_data
import graders as graders_module
from models import (
    ActionType,
    EnvState,
    StepResult,
    TicketObservation,
    TriageAction,
)
from tasks import get_task


class SupportDeskEnv:
    """
    Customer Support Ticket Triage environment.

    OpenEnv interface:
      reset()       -> TicketObservation
      step(action)  -> StepResult(observation, reward, done, info)
      state()       -> EnvState

    Reward shaping:
      - Correct intermediate action -> small positive reward (progress signal)
      - Invalid action              -> -0.05 penalty
      - submit                      -> final graded score as the reward
      - max_steps timeout           -> final score x 0.8 (timeout penalty)
    """

    def __init__(self, task_name: str = "classify", seed: Optional[int] = None):
        valid = ("classify", "triage", "resolve")
        if task_name not in valid:
            raise ValueError(f"Unknown task: {task_name!r}. Valid: {valid}")

        self.task_name = task_name
        self.task      = get_task(task_name)

        # Internal state — set by reset()
        self._ticket:        Dict[str, Any]       = {}
        self._step:          int                  = 0
        self._done:          bool                 = False
        self._actions_taken: List[Dict[str, Any]] = []
        self._total_reward:  float                = 0.0
        self._last_scores:   Dict[str, float]     = {}

        # Ticket pool for this task — rotated deterministically
        self._ticket_pool: List[str] = list(ticket_data.TASK_TICKETS[task_name])
        self._pool_index:  int       = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self) -> TicketObservation:
        """Start a fresh episode. Returns the initial observation."""
        self._done          = False
        self._step          = 0
        self._actions_taken = []
        self._total_reward  = 0.0
        self._last_scores   = {}

        tid = self._ticket_pool[self._pool_index % len(self._ticket_pool)]
        self._pool_index += 1
        self._ticket = ticket_data.TICKET_BY_ID[tid]

        return self._make_obs()

    def step(self, action: TriageAction) -> StepResult:
        """Submit one action. Returns (observation, reward, done, info)."""
        if self._done:
            raise RuntimeError(
                "Episode is already done. Call reset() to start a new episode."
            )

        self._step += 1
        reward               = 0.0
        info: Dict[str, Any] = {}

        # Validate
        error = self._validate(action)
        if error:
            info["error"] = error
            return StepResult(
                observation=self._make_obs(),
                reward=-0.05,
                done=False,
                info=info,
            )

        # Record
        action_dict = action.model_dump(exclude_none=True)
        self._actions_taken.append(action_dict)

        # Intermediate reward: improvement in component scores
        if action.action_type != ActionType.SUBMIT:
            scores    = graders_module.grade(self.task_name, self._actions_taken, self._ticket)
            prev_sum  = sum(v for k, v in self._last_scores.items() if k != "total")
            curr_sum  = sum(v for k, v in scores.items()       if k != "total")
            reward    = max(0.0, round((curr_sum - prev_sum) * 0.4, 4))
            self._last_scores = scores
            info["partial_scores"] = scores

        # Final grading on SUBMIT
        if action.action_type == ActionType.SUBMIT:
            final      = graders_module.grade(self.task_name, self._actions_taken, self._ticket)
            reward     = final["total"]
            self._done = True
            info["final_scores"]    = final
            info["ground_truth"]    = {
                "category": self._ticket["gt_category"],
                "priority": self._ticket["gt_priority"],
                "team":     self._ticket["gt_team"],
            }
            info["missing_actions"] = self._missing()

        # Timeout — max steps reached without submit
        if self._step >= self.task["max_steps"] and not self._done:
            final      = graders_module.grade(self.task_name, self._actions_taken, self._ticket)
            reward     = round(final["total"] * 0.8, 4)
            self._done = True
            info["timeout"]      = True
            info["final_scores"] = final

        self._total_reward = round(self._total_reward + reward, 4)

        return StepResult(
            observation=self._make_obs(),
            reward=round(reward, 4),
            done=self._done,
            info=info,
        )

    def state(self) -> EnvState:
        """Full internal state — useful for debugging and evaluation harnesses."""
        return EnvState(
            task_name     = self.task_name,
            ticket_id     = self._ticket.get("ticket_id", ""),
            step          = self._step,
            done          = self._done,
            total_reward  = self._total_reward,
            actions_taken = list(self._actions_taken),
            solution      = {
                "category": self._ticket.get("gt_category"),
                "priority": self._ticket.get("gt_priority"),
                "team":     self._ticket.get("gt_team"),
            } if self._done else None,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _make_obs(self) -> TicketObservation:
        return TicketObservation(
            ticket_id        = self._ticket.get("ticket_id", ""),
            subject          = self._ticket.get("subject", ""),
            body             = self._ticket.get("body", ""),
            sender_email     = self._ticket.get("sender_email", ""),
            timestamp        = self._ticket.get("timestamp", ""),
            task_name        = self.task_name,
            task_description = self.task["description"],
            current_step     = self._step,
            max_steps        = self.task["max_steps"],
            actions_taken    = list(self._actions_taken),
            current_score    = self._last_scores.get("total", 0.0),
            required_actions = self.task["required_actions"],
            hint             = self.task.get("hint"),
        )

    def _validate(self, action: TriageAction) -> Optional[str]:
        atype = action.action_type
        if atype == ActionType.CLASSIFY and action.category is None:
            return "classify requires the 'category' field"
        if atype == ActionType.SET_PRIORITY and action.priority is None:
            return "set_priority requires the 'priority' field"
        if atype == ActionType.ROUTE and action.team is None:
            return "route requires the 'team' field"
        if atype == ActionType.DRAFT_RESPONSE and not action.response_draft:
            return "draft_response requires the 'response_draft' field"
        # Prevent duplicate actions (draft_response can be revised)
        existing = {a["action_type"] for a in self._actions_taken}
        if atype.value in existing and atype != ActionType.DRAFT_RESPONSE:
            return f"Action '{atype.value}' was already taken this episode"
        return None

    def _missing(self) -> List[str]:
        taken = {a["action_type"] for a in self._actions_taken}
        return [r for r in self.task["required_actions"]
                if r not in taken and r != "submit"]
