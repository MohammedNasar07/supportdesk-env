def format_reward(value: float) -> str:
    """Format reward to exactly 2 decimal places, guaranteed in (0,1)."""
    clamped = max(0.01, min(0.99, float(value)))
    return f"{clamped:.2f}"

def clean_text(text: str) -> str:
    """Aggressive newline and whitespace scrubbing for OpenEnv logging."""
    if not text:
        return "null"
    return str(text).replace("\n", " ").replace("\r", " ").replace("  ", " ").strip()
