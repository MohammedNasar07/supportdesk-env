from __future__ import annotations
import os
import json
import yaml
import importlib
from typing import Any, Dict, List, Optional
from .models import ActionType, StepResult, TicketObservation, TriageAction

SAFE_EPS = 0.0001

class SupportDeskEnv:
    def __init__(self, task_name: str = "classify", seed: Optional[int] = None):
        self.task_name = task_name
        self._load_config()
        self._load_assets()
        
        if task_name not in self.tasks:
            raise ValueError(f"Unknown task: {task_name}")
            
        self.task = self.tasks[task_name]
        self._ticket_pool = list(self.task_mappings.get(task_name, []))
        if not self._ticket_pool:
            # Fallback for safety
            self._ticket_pool = [t["ticket_id"] for t in self.tickets_data[:5]]
            
        self._pool_index = 0
        self.reset()

    def _load_config(self):
        config_path = os.path.join("configs", "tasks.yaml")
        with open(config_path, "r") as f:
            self.tasks = yaml.safe_load(f)

    def _load_assets(self):
        assets_path = os.path.join("assets", "tickets.json")
        with open(assets_path, "r") as f:
            data = json.load(f)
            self.tickets_data = data["tickets"]
            self.task_mappings = data["task_mappings"]
            self.ticket_by_id = {t["ticket_id"]: t for t in self.tickets_data}

    def _get_grader(self):
        # Dynamically load the grader for the current task
        try:
            module = importlib.import_module(f"src.graders.{self.task_name}")
            return module.grade
        except (ImportError, AttributeError):
            # Fallback logic if needed, but should be there
            raise ImportError(f"Could not find grader for task: {self.task_name}")

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
        self._ticket = self.ticket_by_id[tid]
        return self._make_obs()

    def step(self, action: TriageAction) -> StepResult:
        if self._done:
            raise RuntimeError("Episode done.")
        self._step += 1
        info = {}

        # Validation safety (trivial placeholder for now)
        error = self._validate(action)
        if error:
            info["error"] = error
            return StepResult(observation=self._make_obs(), reward=SAFE_EPS, done=False, info=info)

        self._actions_taken.append(action.model_dump(exclude_none=True))
        grade_fn = self._get_grader()

        if action.action_type != ActionType.SUBMIT:
            scores = grade_fn(self._actions_taken, self._ticket)
            # Simple reward shaping: incremental total score
            curr_total = scores.get("total", 0.0)
            prev_total = self._last_scores.get("total", 0.0)
            reward = max(SAFE_EPS, round(curr_total - prev_total, 6))
            self._last_scores = scores
            info["partial_scores"] = scores

        if (action.action_type == ActionType.SUBMIT or self._step >= self.task.get("max_steps", 10)):
            final = grade_fn(self._actions_taken, self._ticket)
            reward = final["total"] if action.action_type == ActionType.SUBMIT else max(SAFE_EPS, round(final["total"] * 0.8, 6))
            self._done = True
            info["final_scores"] = final

        return StepResult(observation=self._make_obs(), reward=reward, done=self._done, info=info)

    def _make_obs(self) -> TicketObservation:
        return TicketObservation(
            ticket_id=self._ticket.get("ticket_id", ""),
            subject=self._ticket.get("subject", ""),
            body=self._ticket.get("body", ""),
            sender_email=self._ticket.get("sender_email", ""),
            timestamp=self._ticket.get("timestamp", ""),
            task_name=self.task_name,
            task_description=self.task.get("description", ""),
            current_step=self._step,
            max_steps=self.task.get("max_steps", 10),
            actions_taken=list(self._actions_taken),
            current_score=self._last_scores.get("total", SAFE_EPS),
            required_actions=self.task.get("required_actions", []),
            hint=self.task.get("hint", None),
        )

    def _validate(self, action: TriageAction) -> Optional[str]:
        return None
