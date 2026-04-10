FORBIDDEN_PHRASES = [
    "refund guaranteed",
    "account restored immediately",
    "unauthorized access is fine",
    "i will give you a refund now",
]

SECURITY_KEYWORDS = [
    "hack",
    "fraud",
    "unauthorized",
    "security",
    "compromised",
    "stolen",
]

def policy_check(ticket_text: str, agent_response: str, escalation: bool) -> float:
    """
    Returns a score from 0.01 to 0.99 for policy compliance.
    """
    resp = agent_response.lower()
    
    # Check for forbidden phrases
    for phrase in FORBIDDEN_PHRASES:
        if phrase in resp:
            return 0.01
            
    # Check for security keywords in ticket vs escalation decision
    ticket = ticket_text.lower()
    has_security_keyword = any(k in ticket for k in SECURITY_KEYWORDS)
    
    if has_security_keyword and not escalation:
        return 0.15
        
    return 0.99
