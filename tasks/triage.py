from openenv import Task

class TriageTask(Task):
    name = "triage"
    max_steps = 3
    description = "Assign correct priority and team to the ticket"
