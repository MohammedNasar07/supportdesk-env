import os
import json
import sys
import requests
import traceback
from openai import OpenAI
from src.utils import format_reward, clean_text

def run_task(task_id: str, client: OpenAI, model_name: str, env_base_url: str):
    """Run a single task iteration with strict logging."""
    steps = 0
    rewards = []
    success = False
    
    # 1. Start Log (Exactly as per benchmark guidance)
    print(f"[START] task={task_id} env=supportflow-arena model={model_name}", flush=True)
    
    try:
        # 2. Reset (Calling the FastAPI server)
        resp = requests.post(f"{env_base_url}/reset", params={"task": task_id})
        resp.raise_for_status()
        obs = resp.json().get("observation", "")
        
        # 3. LLM Action
        system_prompt = "You are a support agent. Return only valid JSON with: category, priority, needs_clarification, escalation, response."
        ai_resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
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
            action_dict = {
                "category": "general", 
                "priority": "low", 
                "needs_clarification": True, 
                "escalation": False, 
                "response": clean_text(content)
            }
            
        # 4. Step
        step_resp = requests.post(f"{env_base_url}/step", json={"message": json.dumps(action_dict)})
        step_resp.raise_for_status()
        data = step_resp.json()
        
        reward = data.get("reward", 0.01)
        done = data.get("done", True)
        steps += 1
        rewards.append(reward)
        
        # 5. Step Log
        action_str = clean_text(json.dumps(action_dict))
        print(f"[STEP] step={steps} action={action_str} reward={format_reward(reward)} done={str(done).lower()} error=null", flush=True)
        
        success = True

    except Exception as e:
        error_msg = clean_text(str(e))
        steps += 1
        # [STEP] must still print so the harness records the failure
        print(f"[STEP] step={steps} action=error reward=0.01 done=true error={error_msg}", flush=True)
        # Debugging goes to stderr ONLY
        print(f"DEBUG Error running task {task_id}: {str(e)}", file=sys.stderr)
        # traceback.print_exc(file=sys.stderr)
        success = False

    finally:
        # 6. End Log (MANDATORY: Guaranteed single print even on exception)
        rewards_list = ",".join(format_reward(r) for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_list}", flush=True)

def main():
    # ─── CONFIGURATION (Loaded inside main to avoid import-time crashes) ───
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    env_base_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

    if hf_token is None:
        raise ValueError("HF_TOKEN environment variable is required")

    # Initialize client
    client = OpenAI(base_url=api_base_url, api_key=hf_token)

    # MANDATORY: Run at least 3 tasks to satisfy Phase 2 Task Validation
    tasks = ["classify", "triage", "resolve"]
    
    for task_id in tasks:
        run_task(task_id, client, model_name, env_base_url)

if __name__ == "__main__":
    main()
