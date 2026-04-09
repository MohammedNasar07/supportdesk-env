#!/usr/bin/env python3

import os
import json
import requests
from typing import List, Dict, Any
from openai import OpenAI

# ───────── CONFIG ─────────

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]  # MUST use this (not HF_TOKEN)

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASKS = ["classify", "triage", "resolve"]  # MUST be 3+

MAX_STEPS = 8
TEMPERATURE = 0.3
MAX_TOKENS = 120

EPS = 1e-6


# ───────── SAFE SCORE (CRITICAL) ─────────


def safe_score(x: Any) -> float:
    try:
        x = float(x)
    except:
        x = 0.0

    if x <= 0.0:
        return EPS
    if x >= 1.0:
        return 1 - EPS
    return x


# ───────── ENV CLIENT ─────────


class EnvClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def reset(self, task: str, seed: int = 42):
        r = requests.post(
            f"{self.base}/reset",
            params={"task": task, "seed": seed},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict):
        r = requests.post(
            f"{self.base}/step",
            json=action,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()


# ───────── LOGGING ─────────


def log_start(task):
    print(f"[START] task={task} env=supportdesk-env model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done, error):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.6f} rewards={rewards_str}",
        flush=True,
    )


# ───────── LLM CALL ─────────


def call_llm(client: OpenAI, text: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful support agent."},
                {"role": "user", "content": text},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return (resp.choices[0].message.content or "ok").strip()
    except:
        return "ok"


# ───────── RUN ONE TASK ─────────


def run_task(client, env, task):

    log_start(task)

    rewards: List[float] = []
    steps = 0
    done = False
    score = EPS

    try:
        obs = env.reset(task=task, seed=42)
        state = obs.get("observation", obs)

        result = {}

        for step in range(1, MAX_STEPS + 1):

            if done:
                break

            action_text = call_llm(client, str(state))

            result = env.step({"message": action_text})

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            state = result.get("observation", state)

            rewards.append(reward)
            steps = step

            log_step(step, action_text, reward, done, result.get("error"))

            if done:
                break

        # ─── FINAL SCORE FIX ───
        info = result.get("info", {})
        final = info.get("final_scores", {})

        if isinstance(final, dict) and final:
            raw = sum(float(v) for v in final.values()) / len(final)
        else:
            raw = sum(rewards) / max(len(rewards), 1)

        score = safe_score(raw)

    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)

    success = score > 0.1

    log_end(success, steps, score, rewards)


# ───────── MAIN ─────────


def main():

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)  # MUST use proxy

    env = EnvClient(ENV_BASE_URL)

    for task in TASKS:
        run_task(client, env, task)


if __name__ == "__main__":
    main()
