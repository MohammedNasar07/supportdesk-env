from openenv import Grader
from graders import grade  # imports from root-level graders.py

class ClassifyGrader(Grader):
    def grade(self, actions, ticket):
        # ticket in OpenEnv might be a dict or object
        # internal grade expects (task_type, actions, ticket)
        score = grade("classify", actions, ticket)["total"]
        return max(0.01, min(0.99, score))
