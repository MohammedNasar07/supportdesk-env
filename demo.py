from inference import infer_action
from src.generator import generate_tickets
from src.grader import grade
from src.schemas import AgentAction

def main():
    tickets = generate_tickets()

    for ticket in tickets[:5]:
        print("\n========================")
        print("Ticket ID:", ticket.ticket_id)
        print("Message:", ticket.customer_message)

        action_dict = infer_action(ticket.customer_message)
        action = AgentAction(**action_dict)

        result = grade(ticket, action)

        print("Agent Action:", action_dict)
        print("Score:", result["total_score"])

if __name__ == "__main__":
    main()
