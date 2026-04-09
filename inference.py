#!/usr/bin/env python3
import os
import requests
import json
from typing import List, Dict, Any
from openai import OpenAI

# ───────── CONFIG ─────────
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# The validator explicitly requires API_KEY to override HF_TOKEN
API_KEY_ENV = os.environ.get("API_KEY")
ACTIVE_KEY = API_KEY_ENV if API_KEY_ENV else os.environ.get("HF_TOKEN")

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860")
TASKS = ["classify", "triage", "resolve"]
BENCHMARK = "supportdesk-env"

MAX_STEPS = 8
TEMPERATURE = 0.3
MAX_TOKENS = 150


# ───────── SAFE SCORE ─────────
def safe_score(x: Any) -> float:
    try:
        val = float(x)
    except Exception:
        return 0.01

    if val <= 0.01:
        return 0.01
    if val >= 0.99:
        return 0.99

    return round(val, 4)


# ───────── ENV CLIENT ─────────
class EnvClient:
    def __init__(self, base_url: str):
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


# ───────── LOGGING ─────────
def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Any) -> None:
    # CRITICAL FIX: Ensure no newlines break the stdout format expected by the validator
    action_clean = str(action).replace("\n", " ").replace("\r", "")
    err_clean = str(error).replace("\n", " ").replace("\r", "") if error else "null"
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={str(done).lower()} error={err_clean}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ───────── LLM ─────────
SYSTEM_PROMPT = "You are a helpful support agent. Respond with valid JSON only."


def call_llm(client: OpenAI, obs: Any) -> str:
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
        text = r.choices[0].message.content
        # Safe default to avoid environment parsing crashes
        return text.strip() if text else '{"action_type": "submit"}'
    except Exception as e:
        return '{"action_type": "submit"}'


# ───────── RUN ONE TASK ─────────
def run_task(client: OpenAI, env: EnvClient, task: str) -> None:
    log_start(task)

    rewards = []
    steps = 0
    done = False
    score = 0.01
    success = False

    try:
        obs = env.reset(task=task, seed=42)
        state = obs.get("observation", obs)
        last_result = {}

        for step in range(1, MAX_STEPS + 1):
            action_text = call_llm(client, state)

            # Wrap in the format the environment expects
            action_payload = {"message": action_text}

            last_result = env.step(action_payload)

            raw_reward = last_result.get("reward")
            reward = float(raw_reward) if raw_reward is not None else 0.01
            done = bool(last_result.get("done", False))
            state = last_result.get("observation", state)
            error = last_result.get("info", {}).get("error")

            rewards.append(reward)
            steps = step

            log_step(step, action_text, reward, done, error)

            if done:
                break

        # score extraction
        info = last_result.get("info", {})
        final = info.get("final_scores", {})

        if final:
            raw_score = sum(float(v) for v in final.values()) / len(final)
        else:
            raw_score = sum(rewards) / max(len(rewards), 1)

        score = safe_score(raw_score)
        success = score > 0.1

    except Exception as e:
        # Guarantee run completion instead of crashing the process
        print(f"[DEBUG] Execution Error: {e}", flush=True)
        score = 0.01
        success = False

    finally:
        log_end(success, steps, score, rewards)


# ───────── MAIN ─────────
def run() -> None:
    if not API_BASE_URL or not ACTIVE_KEY:
        print("[DEBUG] Missing API_BASE_URL or API_KEY", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=ACTIVE_KEY)
    env = EnvClient(ENV_BASE_URL)

    for task in TASKS:
        run_task(client, env, task)


if __name__ == "__main__":
    run()
