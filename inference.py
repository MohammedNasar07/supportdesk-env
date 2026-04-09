#!/usr/bin/env python3
import os
import requests
import json
from typing import List, Dict, Any
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860")

SAFE_EPS = 0.0001


def safe_score(x: Any) -> float:
    try:
        v = float(x)
    except:
        v = 0.5
    return min(max(v, SAFE_EPS), 0.9999)


def log_start(task):
    print(f"[START] task={task} env=supportdesk-env model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done, error):
    # Use 6 decimal places to prevent rounding to 0.00
    a_str = json.dumps(action) if isinstance(action, dict) else str(action)
    print(
        f"[STEP] step={step} action={a_str} reward={reward:.6f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    r_str = ",".join(f"{r:.6f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.6f} rewards={r_str}",
        flush=True,
    )


def run_task(client, task):
    log_start(task)
    rewards, steps, done, score, success = [], 0, False, SAFE_EPS, False
    last_res = {}
    try:
        r_reset = requests.post(
            f"{ENV_BASE_URL}/reset", params={"task": task}, timeout=30
        )
        state = r_reset.json().get("observation", r_reset.json())
        for step in range(1, 11):
            if done:
                break
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": str(state)}],
                temperature=0.1,
            )
            txt = (resp.choices[0].message.content or "").strip()
            r_step = requests.post(
                f"{ENV_BASE_URL}/step",
                json={"message": txt, "action_type": "respond"},
                timeout=30,
            )
            last_res = r_step.json()
            reward = safe_score(last_res.get("reward", SAFE_EPS))
            done, state = bool(last_res.get("done", False)), last_res.get(
                "observation", state
            )
            rewards.append(reward)
            steps = step
            log_step(step, {"message": txt}, reward, done, last_res.get("error"))
        f_total = (
            last_res.get("info", {})
            .get("final_scores", {})
            .get("total", sum(rewards) / len(rewards))
        )
        score, success = safe_score(f_total), safe_score(f_total) > 0.4
    finally:
        log_end(success, steps, score, rewards)


def main():
    if not API_BASE_URL or not API_KEY:
        return
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in ["classify", "triage", "resolve"]:
        run_task(client, task)


if __name__ == "__main__":
    main()
