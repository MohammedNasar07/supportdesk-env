import json
import os
from typing import List
from .schemas import Ticket

def load_tickets(file_path: Optional[str] = None) -> List[Ticket]:
    """
    Load tickets from JSON file and instantiate Ticket objects.
    """
    if file_path is None:
        # Default path
        file_path = os.path.join(os.path.dirname(__file__), "..", "data", "tickets.json")
    
    with open(file_path, "r") as f:
        data = json.load(f)
    
    return [Ticket(**item) for item in data]
