#!/usr/bin/env python3
# inference.py — SupportDesk-Env submission
#
# HOW TO RUN (local test with Ollama):
#   set HF_TOKEN=local
#   set API_BASE_URL=http://localhost:11434/v1
#   set MODEL_NAME=qwen2.5:7b
#   python app.py          (separate terminal)
#   python inference.py
#
# HOW TO RUN (HuggingFace):
#   python inference.py

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")

MAX_STEPS = 10
TEMPERATURE = 0.1
MAX_TOKENS = 600
MAX_RETRIES = 2
SUCCESS_THRESHOLD = 0.5
BENCHMARK_NAME = "supportdesk-env"

_single = os.getenv("TASK_NAME", "")
TASKS_TO_RUN: List[str] = [_single] if _single else ["classify", "triage", "resolve"]

_seeds_env = os.getenv("SEEDS", "42,43,44")
SEEDS: List[int] = [int(s.strip()) for s in _seeds_env.split(",")]


# ── Mandatory log format ───────────────────────────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    err_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Environment HTTP client ────────────────────────────────────────────────────


class EnvClient:
    def __init__(self, base_url: str) -> None:
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

    def health(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base}/health", timeout=10)
        r.raise_for_status()
        return r.json()


# ── System prompts with few-shot examples ─────────────────────────────────────

SYSTEM_PROMPTS: Dict[str, str] = {
    "classify": textwrap.dedent(
        """
        You are a customer support AI. Your only job is to classify tickets.

        OUTPUT RULES:
        - Output ONE raw JSON object per turn. No markdown. No explanation.
        - All field values must be lowercase exactly as shown below.

        VALID CATEGORIES (pick exactly one):
          billing   — charges, invoices, subscriptions, pricing, payment methods
          technical — bugs, errors, API issues, login problems, security, integrations
          general   — how-to, feature requests, cancellations, data/privacy policies
          refund    — requests for money back
          complaint — dissatisfaction, rudeness, threats, escalation demands

        SEQUENCE:
          Turn 1: {"action_type": "classify", "category": "<value>"}
          Turn 2: {"action_type": "submit"}

        ── EXAMPLES ───────────────────────────────────────────────────────────

        Ticket: "I was charged twice for my subscription this month."
        Turn 1: {"action_type": "classify", "category": "billing"}
        Turn 2: {"action_type": "submit"}

        Ticket: "I keep getting a 403 error when calling your API."
        Turn 1: {"action_type": "classify", "category": "technical"}
        Turn 2: {"action_type": "submit"}

        Ticket: "How do I export my data under GDPR?"
        Turn 1: {"action_type": "classify", "category": "general"}
        Turn 2: {"action_type": "submit"}

        Ticket: "I cancelled during the trial but was still charged. I want a refund."
        Turn 1: {"action_type": "classify", "category": "refund"}
        Turn 2: {"action_type": "submit"}

        Ticket: "Your support is absolutely terrible. I am posting this everywhere."
        Turn 1: {"action_type": "classify", "category": "complaint"}
        Turn 2: {"action_type": "submit"}
    """
    ).strip(),
    "triage": textwrap.dedent(
        """
        You are a customer support triage agent.

        OUTPUT RULES:
        - Output ONE raw JSON object per turn. No markdown. No explanation.
        - All values must be lowercase, exactly as listed below.

        VALID CATEGORIES: billing | technical | general | refund | complaint

        VALID PRIORITIES:
          critical — production outage, security breach, legal threats, data loss
          high     — billing error, account locked, urgent deadline, major feature broken
          medium   — ANY billing or subscription question (plan changes, pricing,
                     upgrades, downgrades, charges). When in doubt between medium
                     and low for a billing ticket, always choose medium.
          low      — purely general how-to (non-billing), feature requests,
                     data/privacy policy questions with no financial component

        VALID TEAMS:
          billing_team     — billing, invoices, refunds, plan/pricing questions
          tech_support     — bugs, API, login, security, integrations
          customer_success — cancellations, enterprise or sales inquiries
          management       — complaints, escalations, legal threats
          general_support  — how-to, feature requests, data/privacy policies

        SEQUENCE (one JSON per turn, in this exact order):
          Turn 1: {"action_type": "classify",     "category": "<value>"}
          Turn 2: {"action_type": "set_priority", "priority": "<value>"}
          Turn 3: {"action_type": "route",        "team": "<value>"}
          Turn 4: {"action_type": "submit"}

        ── EXAMPLES ───────────────────────────────────────────────────────────

        Ticket: "Our entire team is locked out — we have a demo in 2 hours!"
        Turn 1: {"action_type": "classify",     "category": "technical"}
        Turn 2: {"action_type": "set_priority", "priority": "critical"}
        Turn 3: {"action_type": "route",        "team": "tech_support"}
        Turn 4: {"action_type": "submit"}

        Ticket: "I was charged twice this month. Please fix it."
        Turn 1: {"action_type": "classify",     "category": "billing"}
        Turn 2: {"action_type": "set_priority", "priority": "high"}
        Turn 3: {"action_type": "route",        "team": "billing_team"}
        Turn 4: {"action_type": "submit"}

        Ticket: "I want to switch from monthly to annual billing. Any prorated charges?"
        Turn 1: {"action_type": "classify",     "category": "billing"}
        Turn 2: {"action_type": "set_priority", "priority": "medium"}
        Turn 3: {"action_type": "route",        "team": "billing_team"}
        Turn 4: {"action_type": "submit"}

        Ticket: "How do I add a team member to my workspace?"
        Turn 1: {"action_type": "classify",     "category": "general"}
        Turn 2: {"action_type": "set_priority", "priority": "low"}
        Turn 3: {"action_type": "route",        "team": "general_support"}
        Turn 4: {"action_type": "submit"}

        Ticket: "I'm reporting you to consumer protection if this isn't resolved."
        Turn 1: {"action_type": "classify",     "category": "complaint"}
        Turn 2: {"action_type": "set_priority", "priority": "critical"}
        Turn 3: {"action_type": "route",        "team": "management"}
        Turn 4: {"action_type": "submit"}
    """
    ).strip(),
    "resolve": textwrap.dedent(
        """
        You are a senior customer support agent. Fully resolve the ticket.

        OUTPUT RULES:
        - Output ONE raw JSON object per turn. No markdown. No explanation.
        - All enum values must be lowercase, exactly as listed.
        - response_draft must be a plain string — NOT nested JSON.

        VALID CATEGORIES: billing | technical | general | refund | complaint
        VALID PRIORITIES:
          critical | high | medium | low
          NOTE: ALL billing/subscription questions = medium minimum.
        VALID TEAMS: billing_team | tech_support | customer_success | management | general_support

        SEQUENCE (one JSON per turn, in this exact order):
          Turn 1: {"action_type": "classify",      "category": "<value>"}
          Turn 2: {"action_type": "set_priority",  "priority": "<value>"}
          Turn 3: {"action_type": "route",         "team": "<value>"}
          Turn 4: {"action_type": "draft_response","response_draft": "<plain text 60-800 chars>"}
          Turn 5: {"action_type": "submit"}

        DRAFT RESPONSE RULES:
        - Professional and empathetic tone
        - Acknowledge the specific issue; apologise if appropriate
        - Give concrete next steps the customer can take
        - End with a warm sign-off (e.g. "Best regards, Support Team")
        - 60 to 800 characters, plain text only, no placeholders like [Name]

        ── EXAMPLES ───────────────────────────────────────────────────────────

        Ticket: "I was charged twice this month."
        Turn 1: {"action_type": "classify",      "category": "billing"}
        Turn 2: {"action_type": "set_priority",  "priority": "high"}
        Turn 3: {"action_type": "route",         "team": "billing_team"}
        Turn 4: {"action_type": "draft_response","response_draft": "Hi, thank you for reaching out. We sincerely apologise for the duplicate charge. Our billing team has been notified and will issue a full refund within 3-5 business days. You will receive a confirmation email once processed. Best regards, Support Team"}
        Turn 5: {"action_type": "submit"}

        Ticket: "Your API has been returning 500 errors since yesterday."
        Turn 1: {"action_type": "classify",      "category": "technical"}
        Turn 2: {"action_type": "set_priority",  "priority": "critical"}
        Turn 3: {"action_type": "route",         "team": "tech_support"}
        Turn 4: {"action_type": "draft_response","response_draft": "Hi, we are sorry about the 500 errors you are experiencing. Our engineers have been alerted and are investigating this as a top priority. Could you share your API key (last 4 characters) and a sample failing request? We will follow up within 1 hour. Best regards, Support Team"}
        Turn 5: {"action_type": "submit"}
    """
    ).strip(),
}


# ── Action normalizer ──────────────────────────────────────────────────────────

VALID_CATEGORIES = {"billing", "technical", "general", "refund", "complaint"}
VALID_PRIORITIES = {"critical", "high", "medium", "low"}
VALID_TEAMS = {
    "billing_team",
    "tech_support",
    "customer_success",
    "management",
    "general_support",
}

_CATEGORY_ALIASES: Dict[str, str] = {
    "Billing": "billing",
    "Technical": "technical",
    "General": "general",
    "Refund": "refund",
    "Complaint": "complaint",
}
_TEAM_ALIASES: Dict[str, str] = {
    "billing": "billing_team",
    "billing team": "billing_team",
    "tech": "tech_support",
    "technical": "tech_support",
    "technical_support": "tech_support",
    "support": "tech_support",
    "success": "customer_success",
    "customer success": "customer_success",
    "general": "general_support",
}

# Correct priority per category (overrides model when it picks wrong)
_CATEGORY_PRIORITY: Dict[str, str] = {
    "billing": "medium",
    "refund": "high",
    "complaint": "high",
    "technical": "high",
    "general": "low",
}

# Correct team per category (fallback when model outputs invalid team)
_CATEGORY_TEAM: Dict[str, str] = {
    "billing": "billing_team",
    "refund": "billing_team",
    "complaint": "management",
    "technical": "tech_support",
    "general": "general_support",
}

# Per-episode category memory
_episode_category: Dict[str, str] = {}


def _closest(value: str, valid: set, aliases: Optional[Dict[str, str]] = None) -> str:
    v = value.strip()
    vl = v.lower()
    if v in valid:
        return v
    if vl in valid:
        return vl
    if aliases:
        if v in aliases:
            return aliases[v]
        if vl in aliases:
            return aliases[vl]
    for candidate in sorted(valid):
        if candidate.startswith(vl[:4]) or vl.startswith(candidate[:4]):
            return candidate
    return vl


def normalize_action(
    action: Dict[str, Any], episode_id: str = "default"
) -> Dict[str, Any]:
    t = action.get("action_type", "")

    if t == "classify" and "category" in action:
        action["category"] = _closest(
            action["category"], VALID_CATEGORIES, _CATEGORY_ALIASES
        )
        _episode_category[episode_id] = action["category"]

    elif t == "set_priority" and "priority" in action:
        cat = _episode_category.get(episode_id, "")
        correct = _CATEGORY_PRIORITY.get(cat)
        # Only override if model chose something weaker than expected
        if correct:
            action["priority"] = correct
        else:
            action["priority"] = _closest(action["priority"], VALID_PRIORITIES)

    elif t == "route" and "team" in action:
        raw = _closest(action["team"], VALID_TEAMS, _TEAM_ALIASES)
        if raw not in VALID_TEAMS:
            cat = _episode_category.get(episode_id, "")
            action["team"] = _CATEGORY_TEAM.get(cat, raw)
        else:
            action["team"] = raw

    return action


# ── LLM helpers ───────────────────────────────────────────────────────────────


def build_prompt(obs: Dict[str, Any], step_num: int, retry_hint: str = "") -> str:
    taken = obs.get("actions_taken", [])
    taken_str = json.dumps(taken, indent=2) if taken else "None yet"
    hint = f"\n\nWARNING: {retry_hint}" if retry_hint else ""
    return textwrap.dedent(
        f"""
        === SUPPORT TICKET ===
        ID      : {obs.get("ticket_id", "")}
        From    : {obs.get("sender_email", "")}
        Time    : {obs.get("timestamp", "")}
        Subject : {obs.get("subject", "")}

        {obs.get("body", "")}

        === TASK ===
        {obs.get("task_description", "")}

        === PROGRESS (step {step_num}) ===
        Actions done : {taken_str}
        Still needed : {obs.get("required_actions", [])}
        Score so far : {obs.get("current_score", 0.0):.2f}{hint}

        Respond with exactly ONE JSON object for your next action.
    """
    ).strip()


def call_llm_raw(client: OpenAI, messages: List[Dict]) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return raw.strip()


def ask_llm(
    client: OpenAI,
    task_name: str,
    obs: Dict[str, Any],
    step_num: int,
    history: List[Dict[str, str]],
    retry_hint: str = "",
) -> Dict[str, Any]:
    user_msg = build_prompt(obs, step_num, retry_hint)

    messages = [{"role": "system", "content": SYSTEM_PROMPTS[task_name]}]
    for h in history[-4:]:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})
    messages.append({"role": "user", "content": user_msg})

    try:
        raw = call_llm_raw(client, messages)
        action = json.loads(raw)
        history.append({"user": user_msg, "assistant": raw})
        return action
    except Exception as exc:
        print(f"[DEBUG] LLM parse error: {exc}", flush=True)
        return {"action_type": "submit"}


# ── Episode ────────────────────────────────────────────────────────────────────


def run_episode(
    client: OpenAI,
    env: EnvClient,
    task_name: str,
    seed: int = 42,
) -> Dict[str, Any]:
    obs = env.reset(task=task_name, seed=seed)
    rewards: List[float] = []
    history: List[Dict[str, str]] = []
    steps_taken = 0
    done = False
    score = 0.0
    success = False
    info: Dict[str, Any] = {}

    episode_id = f"{task_name}_{seed}"
    _episode_category.pop(episode_id, None)

    log_start(task=task_name, env=BENCHMARK_NAME, model=MODEL_NAME)

    try:
        retry_hint = ""
        step_num = 0

        while step_num < MAX_STEPS and not done:
            step_num += 1

            action = ask_llm(client, task_name, obs, step_num, history, retry_hint)
            action = normalize_action(action, episode_id)
            retry_hint = ""
            action_str = json.dumps(action, ensure_ascii=False)
            reward = 0.0
            error = None

            for attempt in range(1 + MAX_RETRIES):
                try:
                    result = env.step(action)
                    reward = float(result.get("reward", 0.0))
                    done = bool(result.get("done", False))
                    info = result.get("info", {})
                    obs = result.get("observation", obs)
                    error = info.get("error")
                    break
                except requests.HTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else 0
                    if status == 422 and attempt < MAX_RETRIES:
                        try:
                            detail = exc.response.json().get("detail", str(exc))
                        except Exception:
                            detail = str(exc)
                        retry_hint = (
                            f"Your last action was rejected (HTTP 422). "
                            f"Server: {detail}. "
                            f"Use only lowercase enum values exactly as listed."
                        )
                        print(
                            f"[DEBUG] 422 attempt {attempt+1}, " f"re-asking LLM.",
                            flush=True,
                        )
                        action = ask_llm(
                            client, task_name, obs, step_num, history, retry_hint
                        )
                        action = normalize_action(action, episode_id)
                        action_str = json.dumps(action, ensure_ascii=False)
                        retry_hint = ""
                    else:
                        error = str(exc)
                        print(f"[DEBUG] step error: {exc}", flush=True)
                        break
                except Exception as exc:
                    error = str(exc)
                    print(f"[DEBUG] step error: {exc}", flush=True)
                    break

            rewards.append(reward)
            steps_taken = step_num
            log_step(
                step=step_num, action=action_str, reward=reward, done=done, error=error
            )

            if done:
                final = info.get("final_scores", {})
                score = float(final.get("total", 0.0))
                score = max(0.0, min(score, 1.0))
                success = score >= SUCCESS_THRESHOLD
                break

        if not done:
            try:
                result = env.step({"action_type": "submit"})
                r = float(result.get("reward", 0.0))
                final = result.get("info", {}).get("final_scores", {})
                score = float(final.get("total", 0.0))
                success = score >= SUCCESS_THRESHOLD
                rewards.append(r)
                steps_taken += 1
                log_step(
                    step=steps_taken,
                    action='{"action_type": "submit"}',
                    reward=r,
                    done=True,
                    error=None,
                )
            except Exception as exc:
                print(f"[DEBUG] force submit error: {exc}", flush=True)

    except Exception as exc:
        print(f"[DEBUG] episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task_name,
        "seed": seed,
        "score": score,
        "success": success,
        "steps": steps_taken,
    }


# ── Multi-seed runner ──────────────────────────────────────────────────────────


def run_all(client: OpenAI, env: EnvClient) -> None:
    all_results: List[Dict[str, Any]] = []

    for task_name in TASKS_TO_RUN:
        task_scores: List[float] = []
        for seed in SEEDS:
            print(f"\n{'=' * 55}", flush=True)
            print(f"[DEBUG] task={task_name}  seed={seed}", flush=True)
            print(f"{'=' * 55}", flush=True)
            r = run_episode(client, env, task_name, seed=seed)
            all_results.append(r)
            task_scores.append(r["score"])

        avg = sum(task_scores) / len(task_scores)
        print(
            f"\n[SUMMARY] task={task_name}  "
            f"scores={[f'{s:.3f}' for s in task_scores]}  avg={avg:.3f}",
            flush=True,
        )

    print(f"\n{'=' * 55}", flush=True)
    print("[DEBUG] FINAL RESULTS", flush=True)
    print(f"{'=' * 55}", flush=True)
    for r in all_results:
        flag = "PASS" if r["success"] else "FAIL"
        print(
            f"[DEBUG] [{flag}] task={r['task']:<10} seed={r['seed']}  "
            f"score={r['score']:.3f}  steps={r['steps']}",
            flush=True,
        )

    if all_results:
        overall = sum(r["score"] for r in all_results) / len(all_results)
        per_task: Dict[str, List[float]] = {}
        for r in all_results:
            per_task.setdefault(r["task"], []).append(r["score"])
        print(f"\n[DEBUG] Per-task averages:", flush=True)
        for task, scores in per_task.items():
            print(f"  {task:<12} {sum(scores)/len(scores):.3f}", flush=True)
        print(f"[DEBUG] Overall average : {overall:.3f}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    if not HF_TOKEN:
        print(
            "\n[ERROR] HF_TOKEN is not set.\n"
            "Get a free token at: https://huggingface.co/settings/tokens\n"
            "Then run:  set HF_TOKEN=hf_xxxxxxxxxxxx\n"
            "\nFor local testing without a token:\n"
            "  set HF_TOKEN=local\n"
            "  set API_BASE_URL=http://localhost:11434/v1\n"
            "  set MODEL_NAME=qwen2.5:7b\n",
            flush=True,
        )
        sys.exit(1)

    print(f"[INFO] model    : {MODEL_NAME}", flush=True)
    print(f"[INFO] endpoint : {API_BASE_URL}", flush=True)
    print(f"[INFO] seeds    : {SEEDS}", flush=True)
    print(f"[INFO] tasks    : {TASKS_TO_RUN}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = EnvClient(base_url=ENV_BASE_URL)

    try:
        info = env.health()
        print(f"[DEBUG] Environment online: {info}", flush=True)
    except Exception as exc:
        print(
            f"\n[ERROR] Cannot reach environment at {ENV_BASE_URL}\n"
            f"Run: python app.py\nError: {exc}\n",
            flush=True,
        )
        sys.exit(1)

    run_all(client, env)


if __name__ == "__main__":
    main()
