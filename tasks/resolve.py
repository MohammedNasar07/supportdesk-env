from openenv import Task

class ResolveTask(Task):
    name = "resolve"
    max_steps = 5
    description = "Draft a correct response and close the ticket"
