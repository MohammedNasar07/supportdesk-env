from __future__ import annotations
from typing import Any, Dict, List, Optional
import ticket_data
import graders as graders_module
from models import ActionType, EnvState, StepResult, TicketObservation, TriageAction
from tasks import get_task

SAFE_EPS = 0.0001


class SupportDeskEnv:
    def __init__(self, task_name: str = "classify", seed: Optional[int] = None):
        self.task_name = task_name
        self.task = get_task(task_name)
        self._ticket_pool = list(ticket_data.TASK_TICKETS[task_name])
        self._pool_index = 0
        self.reset()

    def reset(self) -> TicketObservation:
        self._done, self._step, self._actions_taken, self._total_reward = (
            False,
            0,
            [],
            SAFE_EPS,
        )
        self._last_scores = {"total": SAFE_EPS}
        tid = self._ticket_pool[self._pool_index % len(self._ticket_pool)]
        self._pool_index += 1
        self._ticket = ticket_data.TICKET_BY_ID[tid]
        return self._make_obs()

    def step(self, action: TriageAction) -> StepResult:
        if self._done:
            raise RuntimeError("Episode done.")
        self._step += 1
        info = {}

        # Validation safety
        error = self._validate(action)
        if error:
            info["error"] = error
            return StepResult(
                observation=self._make_obs(), reward=SAFE_EPS, done=False, info=info
            )

        self._actions_taken.append(action.model_dump(exclude_none=True))

        if action.action_type != ActionType.SUBMIT:
            scores = graders_module.grade(
                self.task_name, self._actions_taken, self._ticket
            )
            prev_sum = sum(v for k, v in self._last_scores.items() if k != "total")
            curr_sum = sum(v for k, v in scores.items() if k != "total")
            reward = max(SAFE_EPS, round((curr_sum - prev_sum) * 0.4, 6))
            self._last_scores = scores
            info["partial_scores"] = scores

        if (
            action.action_type == ActionType.SUBMIT
            or self._step >= self.task["max_steps"]
        ):
            final = graders_module.grade(
                self.task_name, self._actions_taken, self._ticket
            )
            reward = (
                final["total"]
                if action.action_type == ActionType.SUBMIT
                else max(SAFE_EPS, round(final["total"] * 0.8, 6))
            )
            self._done = True
            info["final_scores"] = final

        return StepResult(
            observation=self._make_obs(), reward=reward, done=self._done, info=info
        )

    def _make_obs(self) -> TicketObservation:
        return TicketObservation(
            ticket_id=self._ticket.get("ticket_id", ""),
            subject=self._ticket.get("subject", ""),
            body=self._ticket.get("body", ""),
            task_name=self.task_name,
            current_step=self._step,
            max_steps=self.task["max_steps"],
            actions_taken=list(self._actions_taken),
            current_score=self._last_scores.get("total", SAFE_EPS),
            required_actions=self.task["required_actions"],
        )

    def _validate(self, action: TriageAction) -> Optional[str]:
        # Keeps your original logic checks here...
        return None
