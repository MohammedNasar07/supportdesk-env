#!/usr/bin/env python3
# inference.py — FINAL PHASE-2 SAFE VERSION

import os
import requests
from typing import List, Dict, Any

from openai import OpenAI


# ───────────────── CONFIG ─────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")

TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "supportdesk-env")

MAX_STEPS = 8
TEMPERATURE = 0.3
MAX_TOKENS = 150

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


# ───────────────── ENV CLIENT (HTTP ONLY) ─────────────────


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


# ───────────────── LOGGING ─────────────────


def log_start():
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)


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


# ───────────────── LLM ─────────────────

SYSTEM_PROMPT = "You are a helpful agent. Respond with a short message."


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


# ───────────────── MAIN EPISODE ─────────────────


def run():
    if not HF_TOKEN:
        raise RuntimeError("Missing HF_TOKEN / API_KEY")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = EnvClient(ENV_BASE_URL)

    log_start()

    rewards: List[float] = []
    steps = 0
    done = False
    score = EPS
    success = False

    try:
        obs = env.reset(task=TASK_NAME, seed=42)
        state = obs.get("observation", obs)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_text = call_llm(client, state)

            result = env.step({"message": action_text})

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))

            obs = result.get("observation", state)
            state = obs

            rewards.append(reward)
            steps = step

            error = result.get("error", None)

            log_step(step, action_text, reward, done, error)

            if done:
                break

        # ── SAFE SCORE COMPUTATION ──
        info = result.get("info", {}) if isinstance(result, dict) else {}
        final = info.get("final_scores", {}) if isinstance(info, dict) else {}

        if isinstance(final, dict) and len(final) > 0:
            raw = sum(float(v) for v in final.values()) / max(len(final), 1)
        else:
            raw = sum(rewards) / max(len(rewards), 1)

        score = safe_score(raw)
        success = score > 0.1

    except Exception as e:
        # never crash judge
        print(f"[ERROR] {str(e)}", flush=True)

    finally:
        log_end(success, steps, score, rewards)


if __name__ == "__main__":
    run()
