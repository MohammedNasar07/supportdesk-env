import os
import json
import sys
import requests
import traceback
from openai import OpenAI
from src.utils import format_reward, clean_text

# ───────── CONFIG ─────────
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are a customer support triage agent. 
Return ONLY valid JSON including these keys:
- category: billing, account, shipping, technical, security, general
- priority: low, medium, high
- needs_clarification: true or false
- escalation: true or false
- response: a concise and polite customer support reply
"""

def run_task(task_id: str):
    """Run a single task iteration with strict logging."""
    steps = 0
    rewards = []
    success = False
    
    try:
        # 1. Reset
        resp = requests.post(f"{ENV_BASE_URL}/reset", params={"task": task_id})
        resp.raise_for_status()
        obs = resp.json().get("observation", "")
        
        # 2. Start Log
        print(f"[START] task={task_id} env=supportflow-arena model={MODEL_NAME}")
        
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
            action_dict = {"category": "general", "priority": "low", "needs_clarification": True, "escalation": False, "response": clean_text(content)}
            
        # 4. Step
        step_resp = requests.post(f"{ENV_BASE_URL}/step", json={"message": json.dumps(action_dict)})
        step_resp.raise_for_status()
        data = step_resp.json()
        
        reward = data.get("reward", 0.01)
        done = data.get("done", True)
        steps += 1
        rewards.append(reward)
        
        # 5. Step Log
        action_str = clean_text(json.dumps(action_dict))
        print(f"[STEP] step={steps} action={action_str} reward={format_reward(reward)} done={str(done).lower()} error=null")
        
        success = True

    except Exception as e:
        error_msg = clean_text(str(e))
        steps += 1
        # Ensure error [STEP] is still on stdout so the evaluator sees it
        print(f"[STEP] step={steps} action=error reward=0.01 done=true error={error_msg}")
        # Detailed error info goes to stderr only
        print(f"DEBUG: Task {task_id} failed: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    finally:
        # 6. End Log (Strict Format: Guaranteed single print)
        rewards_list = ",".join(format_reward(r) for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_list}")

def main():
    # MANDATORY: Run at least 3 tasks to pass Phase 2 Validator
    tasks = ["classify", "triage", "resolve"]
    for t in tasks:
        run_task(t)

if __name__ == "__main__":
    main()
