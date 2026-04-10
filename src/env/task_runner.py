class TaskRunner:
    def __init__(self, task, grader):
        self.task = task
        self.grader = grader
        self.state = {}

    def run(self, env):
        # Humanized logic for running a task in the environment
        obs = env.reset()
        return {
            "task_name": self.task["name"],
            "success": False,
            "steps_done": [],
        }
