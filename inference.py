#!/usr/bin/env python3
# inference.py — ROBUST MULTI-TASK VERSION (PHASE 2 SAFE)

import os
import json
import time
import requests
from typing import List, Dict, Any, Optional
from openai import OpenAI

# ─── MANDATORY CONFIG ───
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860")

TASKS = ["classify", "triage", "resolve"]
BENCHMARK = "supportdesk-env"

MAX_STEPS = 8
EPS = 0.0001  # Strict safety margin


def safe_score(x: Any) -> float:
    try:
        v = float(x)
        if v != v:
            v = 0.5
    except:
        v = 0.5
    return min(max(v, EPS), 1.0 - EPS)


# ─── ROBUST ENV CLIENT ───


class EnvClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def _call(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base}{path}"
        try:
            r = requests.request(method, url, timeout=45, **kwargs)
            # If we get HTML instead of JSON (common on HF Space wake-up)
            if "application/json" not in r.headers.get("Content-Type", "").lower():
                print(
                    f"[DEBUG] Error: Expected JSON from {path}, got HTML. Space might be sleeping.",
                    flush=True,
                )
                return {"error": "non_json_response", "status": r.status_code}

            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[DEBUG] Request to {path} failed: {e}", flush=True)
            return {"error": str(e), "done": True}

    def reset(self, task: str):
        return self._call("POST", "/reset", params={"task": task, "seed": 42})

    def step(self, action: Dict[str, Any]):
        return self._call("POST", "/step", json=action)


# ─── LOGGING ───


def log_start(task):
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done, error):
    # Action must be a string for the log
    a_str = json.dumps(action) if isinstance(action, dict) else str(action)
    print(
        f"[STEP] step={step} action={a_str} reward={reward:.6f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    rew_str = ",".join(f"{r:.6f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.6f} rewards={rew_str}",
        flush=True,
    )


# ─── RUNNER ───


def run_task(client, env, task):
    log_start(task)
    rewards, steps, done, score, success = [], 0, False, EPS, False
    last_res = {}

    try:
        # 1. Reset (with retry if space is sleeping)
        obs_data = env.reset(task)
        if "error" in obs_data:
            time.sleep(5)
            obs_data = env.reset(task)

        state = obs_data.get("observation", obs_data)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # 2. LLM Call
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": str(state)}],
                    temperature=0.3,
                    max_tokens=150,
                )
                action_text = (resp.choices[0].message.content or "hello").strip()
            except:
                action_text = "Support requested."

            # 3. Env Step
            # Environment expects {"message": ...}
            last_res = env.step({"message": action_text})

            reward = safe_score(last_res.get("reward", EPS))
            done = bool(last_res.get("done", False))
            state = last_res.get("observation", state)

            rewards.append(reward)
            steps = step
            log_step(step, action_text, reward, done, last_res.get("error"))

        # 4. Final Scoring
        info = last_res.get("info", {})
        final_scores = info.get("final_scores", {}) if isinstance(info, dict) else {}

        if final_scores:
            raw = sum(float(v) for v in final_scores.values()) / len(final_scores)
        else:
            raw = sum(rewards) / max(len(rewards), 1)

        score = safe_score(raw)
        success = score > 0.1

    except Exception as e:
        print(f"[DEBUG] Task Error: {e}", flush=True)
    finally:
        log_end(success, steps, score, rewards)


def run():
    if not API_KEY or not API_BASE_URL:
        print("[ERROR] Missing API_BASE_URL or API_KEY", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EnvClient(ENV_BASE_URL)

    for task in TASKS:
        run_task(client, env, task)


if __name__ == "__main__":
    run()
