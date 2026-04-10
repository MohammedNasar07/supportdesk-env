from openenv import Grader
from graders import grade

class ResolveGrader(Grader):
    def grade(self, actions, ticket):
        score = grade("resolve", actions, ticket)["total"]
        return max(0.01, min(0.99, score))
