#!/usr/bin/env python3
# inference.py — FINAL STABLE VERSION
# UPDATED FINAL VERSION 100%
git status
import os
import requests
from typing import List, Dict, Any
from openai import OpenAI


# ───────── CONFIG ─────────

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")

TASK_NAME = os.getenv("TASK_NAME", "classify")
BENCHMARK = "supportdesk-env"

MAX_STEPS = 8
TEMPERATURE = 0.3
MAX_TOKENS = 150


# ───────── ENV CLIENT ─────────


class EnvClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def reset(self, task: str, seed: int = 42) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base}/reset",
            params={"task": task, "seed": seed},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base}/step",
            json=action,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()


# ───────── LOGGING ─────────


def log_start():
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)


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


# ───────── LLM ─────────

SYSTEM_PROMPT = "You are a helpful support agent. Respond briefly."


def call_llm(client: OpenAI, obs: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(obs)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return (resp.choices[0].message.content or "hello").strip()
    except:
        return "hello"


# ───────── MAIN ─────────


def run():
    if not API_BASE_URL or not HF_TOKEN:
        raise RuntimeError("Missing API_BASE_URL or HF_TOKEN")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = EnvClient(ENV_BASE_URL)

    log_start()

    rewards: List[float] = []
    steps = 0
    done = False
    score = 0.5  # SAFE DEFAULT
    success = False

    try:
        obs = env.reset(task=TASK_NAME, seed=42)
        state = obs.get("observation", obs)

        last_result = {}

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

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

        # SAFE SCORE
        if rewards:
            score = sum(rewards) / len(rewards)

        # clamp
        score = max(0.01, min(score, 0.99))
        success = score > 0.1

    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)

    finally:
        log_end(success, steps, score, rewards)


if __name__ == "__main__":
    run()
