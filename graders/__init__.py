from openenv import Task
from .classify_grader import ClassifyGrader
from .triage_grader import TriageGrader
from .resolve_grader import ResolveGrader

Task.register_grader("classify", ClassifyGrader)
Task.register_grader("triage", TriageGrader)
Task.register_grader("resolve", ResolveGrader)
