from src.env.support_env import SupportDeskEnv

def main():
    print("🚀 Running SupportFlow Arena Local Demo")
    tasks = ["classify", "triage", "resolve"]
    
    for task_name in tasks:
        print(f"\n--- Testing Task: {task_name} ---")
        env = SupportDeskEnv(task_name=task_name)
        obs = env.reset()
        print(f"Ticket ID: {obs.ticket_id}")
        print(f"Subject: {obs.subject}")
        print(f"Goal: {obs.task_description}")
        print("Success: Env Loaded and Ready.")

if __name__ == "__main__":
    main()
