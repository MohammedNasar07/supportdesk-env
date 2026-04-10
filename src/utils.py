def format_reward(value: float) -> str:
    """Format reward to exactly 2 decimal places."""
    return f"{value:.2f}"

def clean_text(text: str) -> str:
    """Aggressive newline and whitespace scrubbing for OpenEnv logging."""
    if not text:
        return "null"
    return str(text).replace("\n", " ").replace("\r", " ").replace("  ", " ").strip()
