from openenv import Task

class ClassifyTask(Task):
    name = "classify"
    max_steps = 3
    description = "Classify the support ticket into the correct category"
