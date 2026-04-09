#!/usr/bin/env python3
# inference.py — FINAL MULTI-TASK VALIDATOR SAFE

import os
import requests
from typing import List, Dict, Any
from openai import OpenAI


# ───────── CONFIG ─────────

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")

TASKS = ["classify", "triage", "resolve"]  # ✅ MUST

BENCHMARK = "supportdesk-env"

MAX_STEPS = 8
TEMPERATURE = 0.3
MAX_TOKENS = 150

EPS = 1e-6


# ───────── SAFE SCORE ─────────


def safe_score(x):
    try:
        x = float(x)
    except:
        x = 0.5

    if x <= 0.0:
        return EPS
    if x >= 1.0:
        return 1 - EPS
    return x


# ───────── ENV CLIENT ─────────


class EnvClient:
    def __init__(self, base_url):
        self.base = base_url.rstrip("/")

    def reset(self, task, seed=42):
        r = requests.post(
            f"{self.base}/reset", params={"task": task, "seed": seed}, timeout=30
        )
        r.raise_for_status()
        return r.json()

    def step(self, action):
        r = requests.post(f"{self.base}/step", json=action, timeout=30)
        r.raise_for_status()
        return r.json()


# ───────── LOGGING ─────────


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


# ───────── LLM ─────────

SYSTEM_PROMPT = "You are a helpful support agent."


def call_llm(client, obs):
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(obs)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return (r.choices[0].message.content or "hello").strip()
    except:
        return "hello"


# ───────── RUN ONE TASK ─────────


def run_task(client, env, task):

    log_start(task)

    rewards = []
    steps = 0
    done = False
    score = EPS
    success = False

    try:
        obs = env.reset(task=task, seed=42)
        state = obs.get("observation", obs)

        last_result = {}

        for step in range(1, MAX_STEPS + 1):

            action_text = call_llm(client, state)

            last_result = env.step({"message": action_text})

            reward = float(last_result.get("reward", 0.0))
            done = bool(last_result.get("done", False))

            state = last_result.get("observation", state)

            rewards.append(reward)
            steps = step

            error = last_result.get("error", None)

            log_step(step, action_text, reward, done, error)

            if done:
                break

        # score
        info = last_result.get("info", {})
        final = info.get("final_scores", {})

        if final:
            raw = sum(float(v) for v in final.values()) / len(final)
        else:
            raw = sum(rewards) / max(len(rewards), 1)

        score = safe_score(raw)
        success = score > 0.1

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        score = EPS
        success = False

    finally:
        log_end(success, steps, score, rewards)


# ───────── MAIN ─────────


def run():

    if not API_BASE_URL or not HF_TOKEN:
        raise RuntimeError("Missing API_BASE_URL / HF_TOKEN")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = EnvClient(ENV_BASE_URL)

    # ✅ RUN ALL TASKS
    for task in TASKS:
        run_task(client, env, task)


if __name__ == "__main__":
    run()
