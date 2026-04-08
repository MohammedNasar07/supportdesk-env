# app.py
# SupportDesk-Env FastAPI server.
# Run with:  python app.py
# All imports are direct module names — no packages, no relative imports.

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# These files must be in the SAME folder as app.py
from environment import SupportDeskEnv
from models import TriageAction
from tasks import TASKS

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "SupportDesk-Env",
    description = "OpenEnv customer support ticket triage environment.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Single shared env instance (sequential use)
_env: Optional[SupportDeskEnv] = None


def _get_env() -> SupportDeskEnv:
    global _env
    if _env is None:
        _env = SupportDeskEnv(task_name="classify")
    return _env


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "env": "supportdesk-env", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "name":             t["name"],
                "description":      t["description"],
                "difficulty":       t["difficulty"],
                "required_actions": t["required_actions"],
                "max_steps":        t["max_steps"],
            }
            for t in TASKS.values()
        ]
    }


@app.post("/reset")
def reset(
    task: str = Query(
        default="classify",
        description="Task name: classify | triage | resolve",
    ),
    seed: Optional[int] = Query(
        default=None,
        description="Optional integer seed",
    ),
) -> Dict[str, Any]:
    global _env
    if task not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: {task!r}. Valid: {list(TASKS.keys())}",
        )
    _env = SupportDeskEnv(task_name=task, seed=seed)
    obs  = _env.reset()
    return obs.model_dump()


@app.post("/step")
def step(action: TriageAction) -> Dict[str, Any]:
    env = _get_env()
    try:
        result = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "observation": result.observation.model_dump(),
        "reward":      result.reward,
        "done":        result.done,
        "info":        result.info,
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    return _get_env().state().model_dump()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    print(f"Starting SupportDesk-Env on http://localhost:{port}", flush=True)
    print("Press Ctrl+C to stop.\n", flush=True)
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
