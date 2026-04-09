#!/usr/bin/env python3
# inference.py — FINAL WINNING VERSION (PROXY + HIGH SCORE)

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

TASK_NAME = os.environ.get("MY_ENV_V4_TASK", "classify")
BENCHMARK = os.environ.get("MY_ENV_V4_BENCHMARK", "supportdesk-env")

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


# ───────────────── SMART AGENT (BOOSTED SCORE) ─────────────────

SYSTEM_PROMPT = """
You are a highly accurate customer support AI.

STRICT RULES:
- Always follow required action sequence
- Always use lowercase values
- Output ONLY valid JSON
- No explanations

TASKS:

CLASSIFY:
Turn1: {"action_type":"classify","category":"billing|technical|general|refund|complaint"}
Turn2: {"action_type":"submit"}

TRIAGE:
Turn1: classify
Turn2: set_priority
Turn3: route
Turn4: submit

RESOLVE:
Turn1: classify
Turn2: set_priority
Turn3: route
Turn4: draft_response
Turn5: submit

IMPORTANT:
- billing → billing_team
- technical → tech_support
- complaint → management
- general → general_support
- refund → billing_team
"""


def call_llm(client: OpenAI, obs: Dict[str, Any]) -> Dict[str, Any]:
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

    except Exception:
        return {"action_type": "submit"}


# ───────────────── MAIN LOOP ─────────────────


def run():
    # ✅ PROXY CORRECT
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    # ✅ ENV CORRECT
    env = EnvClient(ENV_BASE_URL)

    log_start()

    rewards: List[float] = []
    steps = 0
    done = False
    score = EPS
    success = False

    try:
        obs = env.reset(task=TASK_NAME, seed=42)

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

            error = result.get("error", None)

            log_step(step, json.dumps(action), reward, done, error)

            if done:
                break

        # ── SCORE ──
        info = result.get("info", {}) if isinstance(result, dict) else {}
        final = info.get("final_scores", {}) if isinstance(info, dict) else {}

        if isinstance(final, dict) and len(final) > 0:
            raw_score = sum(float(v) for v in final.values()) / len(final)
        else:
            raw_score = sum(rewards) / max(len(rewards), 1)

        score = safe_score(raw_score)
        success = score > 0.5

    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)

    finally:
        log_end(success, steps, score, rewards)


if __name__ == "__main__":
    run()
