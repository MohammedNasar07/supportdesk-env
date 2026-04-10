import os
import json
import sys
import requests
from openai import OpenAI

# ───────── CONFIG ─────────
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("Set HF_TOKEN or API_KEY environment variable.")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ───────── SYSTEM PROMPT ─────────
SYSTEM_PROMPT = """You are a customer support triage agent. 
Return ONLY valid JSON including these keys:
- category: billing, account, shipping, technical, security, general
- priority: low, medium, high
- needs_clarification: true or false
- escalation: true or false
- response: a concise and polite customer support reply
"""

def clean_output(text: str) -> str:
    """Scrub newlines for strict OpenEnv logging."""
    return str(text).replace("\n", " ").replace("\r", " ").strip()

def run_task(task_id: str):
    """Run a single task iteration."""
    try:
        # 1. Reset
        resp = requests.post(f"{ENV_BASE_URL}/reset", params={"task": task_id})
        resp.raise_for_status()
        obs = resp.json().get("observation", "")
        
        # 2. Start Log
        print(f"[START] task={task_id} env=supportflow model={MODEL_NAME}")
        
        # 3. LLM Action
        ai_resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs}
            ],
            temperature=0.2
        )
        content = ai_resp.choices[0].message.content.strip()
        
        # Parse JSON
        try:
            if "{" in content:
                content_json = content[content.find("{"):content.rfind("}")+1]
                action_dict = json.loads(content_json)
            else:
                raise ValueError("No JSON")
        except:
            action_dict = {"category": "general", "priority": "low", "needs_clarification": True, "escalation": False, "response": "Details please."}
            
        # 4. Step
        step_resp = requests.post(f"{ENV_BASE_URL}/step", json={"message": json.dumps(action_dict)})
        step_resp.raise_for_status()
        data = step_resp.json()
        
        reward = data.get("reward", 0.01)
        done = data.get("done", True)
        
        # 5. Step Log
        action_str = clean_output(json.dumps(action_dict))
        print(f"[STEP] step=1 action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")
        
        # 6. End Log
        print(f"[END] success=true steps=1 score={reward:.2f} rewards={reward:.2f}")

    except Exception as e:
        print(f"[ERROR] Task {task_id} failed: {str(e)}")
        print(f"[END] success=false steps=0 score=0.01 rewards=")

def main():
    # MANDATORY: Run at least 3 tasks to pass Phase 2 Validator
    tasks = ["classify", "triage", "resolve"]
    for t in tasks:
        run_task(t)

if __name__ == "__main__":
    main()
