from openenv import Grader
from . import grade

class TriageGrader(Grader):
    def grade(self, actions, ticket):
        score = grade("triage", actions, ticket)["total"]
        return max(0.01, min(0.99, score))
