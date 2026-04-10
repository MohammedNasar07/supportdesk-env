import os
import json
import requests
from openai import OpenAI

# ───────── CONFIG ─────────
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

SYSTEM_PROMPT = """You are a customer support triage agent.
Return ONLY valid JSON with these keys:
- category: [billing, account, shipping, technical, security, general]
- priority: [low, medium, high]
- team: [support, finance, engineering]
- response_draft: a short helpful customer support reply
"""

def run_inference(ticket_text: str) -> dict:
    token = os.getenv("HF_TOKEN")
    if not token:
        return {"response_draft": "Error: HF_TOKEN is not set.", "category": "general", "priority": "medium"}

    client = OpenAI(base_url=API_BASE_URL, api_key=token)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ticket_text}
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except:
        return {"response_draft": content, "category": "general", "priority": "medium"}

def main():
    task_id = os.getenv("TASK_ID", "classify")
    print(f"[START] task={task_id} env=supportdesk-env model={MODEL_NAME}")

    try:
        resp = requests.post(f"{ENV_BASE_URL}/reset", params={"task": task_id})
        resp.raise_for_status()
        data = resp.json()
        observation = data.get("observation", "")
        done = data.get("done", False)

        step = 1
        rewards = []

        while not done and step <= 5:
            decision = run_inference(observation)
            action_payload = {"message": json.dumps(decision)}
            
            step_resp = requests.post(f"{ENV_BASE_URL}/step", json=action_payload)
            step_resp.raise_for_status()
            step_data = step_resp.json()
            
            observation = step_data.get("observation", "")
            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", False)
            rewards.append(reward)

            print(f"[STEP] step={step} action=triage reward={reward:.2f} done={str(done).lower()} error=null")
            step += 1

        total_score = sum(rewards) / len(rewards) if rewards else 0.0
        success = total_score > 0.5
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={str(success).lower()} steps={len(rewards)} score={total_score:.6f} rewards={rewards_str}")

    except Exception as e:
        print(f"[END] success=false steps=0 score=0.000000 rewards= error={str(e)}")

if __name__ == "__main__":
    main()
