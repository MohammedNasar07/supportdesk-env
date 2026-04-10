FORBIDDEN_PROMISES = [
    "refund approved",
    "i will refund",
    "refund issued",
    "account restored immediately",
    "guarantee",
]

SECURITY_KEYWORDS = [
    "hack",
    "hacked",
    "fraud",
    "stolen",
    "unauthorized",
    "compromised",
    "breach",
]

AMBIGUOUS_HINTS = [
    "not sure",
    "maybe",
    "i think",
    "could be",
    "don't know",
    "unclear",
]

def is_security_related(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in SECURITY_KEYWORDS)

def is_ambiguous(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in AMBIGUOUS_HINTS)

def has_forbidden_promise(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in FORBIDDEN_PROMISES)
