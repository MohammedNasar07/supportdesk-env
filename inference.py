#!/usr/bin/env python3
# inference.py — SupportDesk-Env submission (FINAL FIXED VERSION)

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI


# ── CONFIG ──────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")

MAX_STEPS = 10
TEMPERATURE = 0.1
MAX_TOKENS = 600

EPS = 1e-6
SUCCESS_THRESHOLD = 0.5

BENCHMARK_NAME = "supportdesk-env"

_single = os.getenv("TASK_NAME", "")
TASKS_TO_RUN: List[str] = [_single] if _single else ["classify", "triage", "resolve"]

_seeds_env = os.getenv("SEEDS", "42,43,44")
SEEDS: List[int] = [int(s.strip()) for s in _seeds_env.split(",")]


# ── SAFE SCORE ─────────────────────────────────────────────────────────


def safe_score(x: Any) -> float:
    """
    STRICT validator-safe clamp into (0,1)
    """
    try:
        v = float(x)
    except Exception:
        v = 0.0

    if v <= 0.0:
        return EPS
    if v >= 1.0:
        return 1 - EPS
    return v


# ── LOGGING ────────────────────────────────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    err_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.6f} rewards={rewards_str}",
        flush=True,
    )


# ── ENV CLIENT ─────────────────────────────────────────────────────────


class EnvClient:
    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")

    def reset(self, task: str, seed: int = 42) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base}/reset", params={"task": task, "seed": seed}, timeout=30
        )
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(f"{self.base}/step", json=action, timeout=30)
        r.raise_for_status()
        return r.json()


# ── PROMPTS ─────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "classify": "You are a support classifier. Output only JSON.",
    "triage": "You are a support triage agent. Output only JSON.",
    "resolve": "You are a senior support agent. Output only JSON.",
}


def build_prompt(obs: Dict[str, Any], step_num: int) -> str:
    return f"""
TICKET:
{obs.get("body","")}

STEP: {step_num}
TASK: {obs.get("task_description","")}

Return ONLY JSON.
""".strip()


def call_llm(client: OpenAI, messages: List[Dict]) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return (resp.choices[0].message.content or "").strip()


def ask_llm(
    client: OpenAI, task: str, obs: Dict[str, Any], step: int
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[task]},
        {"role": "user", "content": build_prompt(obs, step)},
    ]

    try:
        raw = call_llm(client, messages)

        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()

        return json.loads(raw)

    except Exception:
        return {"action_type": "submit"}


# ── EPISODE ─────────────────────────────────────────────────────────────


def run_episode(client: OpenAI, env: EnvClient, task: str, seed: int):
    obs = env.reset(task=task, seed=seed)

    rewards: List[float] = []
    steps = 0
    done = False

    score = EPS
    success = False

    log_start(task, BENCHMARK_NAME, MODEL_NAME)

    while steps < MAX_STEPS and not done:
        steps += 1

        action = ask_llm(client, task, obs, steps)

        reward = 0.0
        error = None

        try:
            result = env.step(action)

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            obs = result.get("observation", obs)

            info = result.get("info", {})
            final = info.get("final_scores", {})

            # ✅ SAFE SCORE (CRITICAL FIX)
            raw_score = final.get("total", 0.0)
            if raw_score is None:
                raw_score = 0.0

            score = safe_score(raw_score)

        except Exception as e:
            error = str(e)

        rewards.append(reward)

        log_step(steps, json.dumps(action), reward, done, error)

        if done:
            break

    # ── FINAL SUBMIT SAFETY ────────────────────────────────
    if not done:
        try:
            result = env.step({"action_type": "submit"})

            final = result.get("info", {}).get("final_scores", {})

            raw_score = final.get("total", 0.0)
            if raw_score is None:
                raw_score = 0.0

            score = safe_score(raw_score)

            rewards.append(float(result.get("reward", 0.0)))
            steps += 1

            log_step(steps, '{"action_type":"submit"}', rewards[-1], True, None)

        except Exception:
            pass

    # FINAL SAFETY (DOUBLE PROTECTION)
    score = safe_score(score)

    success = score > SUCCESS_THRESHOLD

    log_end(success, steps, score, rewards)

    return {
        "task": task,
        "seed": seed,
        "score": score,
        "success": success,
        "steps": steps,
    }


# ── RUNNER ─────────────────────────────────────────────────────────────


def run_all(client: OpenAI, env: EnvClient):
    results = []

    for task in TASKS_TO_RUN:
        scores = []

        for seed in SEEDS:
            r = run_episode(client, env, task, seed)
            results.append(r)
            scores.append(r["score"])

        print(f"[SUMMARY] {task} avg={sum(scores)/len(scores):.4f}")

    print("\n[FINAL]")
    for r in results:
        print(r)


# ── MAIN ───────────────────────────────────────────────────────────────


def main():
    if not HF_TOKEN:
        print("HF_TOKEN missing")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = EnvClient(ENV_BASE_URL)

    run_all(client, env)


if __name__ == "__main__":
    main()
