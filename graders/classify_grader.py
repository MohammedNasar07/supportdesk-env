from openenv import Grader
from . import grade  # imports from __init__.py

class ClassifyGrader(Grader):
    def grade(self, actions, ticket):
        # ticket in OpenEnv might be a dict or object
        # internal grade expects (task_type, actions, ticket)
        score = grade("classify", actions, ticket)["total"]
        return max(0.01, min(0.99, score))
