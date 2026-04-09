#!/usr/bin/env python3
# inference.py — FINAL 100% PASS (MULTI-TASK + PROXY + SAFE SCORE)

import os
import json
import requests
from typing import List, Dict, Any
from openai import OpenAI


# ───────────────── CONFIG ─────────────────

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860")

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASKS = ["classify", "triage", "resolve"]
BENCHMARK = "supportdesk-env"

MAX_STEPS = 10
TEMPERATURE = 0.2
MAX_TOKENS = 200

EPS = 1e-6


# ───────────────── SAFE SCORE ─────────────────


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


# ───────────────── ENV CLIENT ─────────────────


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

    def step(self, action: Dict[str, Any]):
        r = requests.post(f"{self.base}/step", json=action, timeout=30)
        r.raise_for_status()
        return r.json()


# ───────────────── LOGGING ─────────────────


def log_start(task):
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done, error):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.6f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ───────────────── SMART AGENT ─────────────────

SYSTEM_PROMPT = """
You are a strict support AI.

RULES:
- Output ONLY JSON
- lowercase values
- follow correct sequence

TASKS:

classify:
1 classify
2 submit

triage:
1 classify
2 set_priority
3 route
4 submit

resolve:
1 classify
2 set_priority
3 route
4 draft_response
5 submit
"""


def call_llm(client, obs):
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(obs)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        text = (resp.choices[0].message.content or "").strip()

        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        return json.loads(text)

    except:
        return {"action_type": "submit"}


# ───────────────── RUN ONE TASK ─────────────────


def run_task(client, env, task):

    log_start(task)

    rewards = []
    steps = 0
    done = False
    score = EPS

    try:
        obs = env.reset(task=task, seed=42)
        result = {}

        for step in range(1, MAX_STEPS + 1):

            if done:
                break

            action = call_llm(client, obs)
            result = env.step(action)

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))

            obs = result.get("observation", obs)

            rewards.append(reward)
            steps = step

            log_step(step, json.dumps(action), reward, done, None)

        info = result.get("info", {})
        final = info.get("final_scores", {})

        if isinstance(final, dict) and len(final) > 0:
            raw_score = sum(float(v) for v in final.values()) / len(final)
        else:
            raw_score = sum(rewards) / max(len(rewards), 1)

        score = safe_score(raw_score)
        success = score > 0.5

    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)
        success = False

    log_end(success, steps, score, rewards)


# ───────────────── MAIN ─────────────────


def run():

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    env = EnvClient(ENV_BASE_URL)

    for task in TASKS:
        run_task(client, env, task)


if __name__ == "__main__":
    run()
