#!/usr/bin/env python3
# inference.py — MINIMAL SUBMISSION SAFE VERSION

import os
import asyncio
from typing import List, Optional

from openai import OpenAI
from my_env_v4 import MyEnvV4Action, MyEnvV4Env

# ── CONFIG ───────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

IMAGE_NAME = os.getenv("IMAGE_NAME")

TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")

MAX_STEPS = 8
TEMPERATURE = 0.5
MAX_TOKENS = 150

EPS = 1e-6


# ── SAFE SCORE ───────────────────────────────────────────


def safe_score(x: float) -> float:
    try:
        x = float(x)
    except:
        x = 0.0

    if x <= 0.0:
        return EPS
    if x >= 1.0:
        return 1 - EPS
    return x


# ── LOGGING ──────────────────────────────────────────────


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


# ── LLM ───────────────────────────────────────────────────

SYSTEM_PROMPT = "You are an assistant. Reply with a short message only."


def get_action(client: OpenAI, obs: str) -> str:
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return (r.choices[0].message.content or "hello").strip()
    except:
        return "hello"


# ── MAIN EPISODE ─────────────────────────────────────────


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)

    rewards: List[float] = []
    steps = 0
    score = EPS
    success = False

    log_start()

    try:
        result = await env.reset()
        obs = result.observation.echoed_message
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = get_action(client, obs)

            result = await env.step(MyEnvV4Action(message=action))

            reward = float(result.reward or 0.0)
            done = bool(result.done)

            obs = result.observation.echoed_message

            rewards.append(reward)
            steps = step

            log_step(step, action, reward, done, None)

            if done:
                break

        # ── SAFE SCORE COMPUTATION ──
        info = result.get("info", {}) if isinstance(result, dict) else {}
        final = info.get("final_scores", {}) if isinstance(info, dict) else {}

        if isinstance(final, dict) and len(final) > 0:
            raw_score = sum(float(v) for v in final.values()) / max(len(final), 1)
        else:
            raw_score = sum(rewards) / max(len(rewards), 1)

        score = safe_score(raw_score)
        success = score > 0.1

    finally:
        try:
            await env.close()
        except:
            pass

        log_end(success, steps, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())
