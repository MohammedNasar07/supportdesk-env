import os
import sys
import importlib.util
import traceback
from pathlib import Path

ROOT = Path(".").resolve()

REQUIRED_FILES = [
    "inference.py",
    "requirements.txt",
    "README.md",
]

def check_files():
    missing = [f for f in REQUIRED_FILES if not (ROOT / f).exists()]
    if missing:
        return False, f"Missing required files: {missing}"
    return True, "Required files present"

def check_env_vars():
    api_base = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    hf_token = os.getenv("HF_TOKEN")

    if not api_base:
        return False, "API_BASE_URL invalid"
    if not model_name:
        return False, "MODEL_NAME invalid"
    if hf_token is None:
        return False, "HF_TOKEN missing"
    return True, "Environment variables look good"

def import_inference():
    spec = importlib.util.spec_from_file_location("inference", str(ROOT / "inference.py"))
    module = importlib.util.module_from_spec(spec)
    try:
        if spec.loader:
            spec.loader.exec_module(module)
        else:
            return False, "Loader missing"
    except Exception as e:
        return False, f"inference.py import failed: {e}"
    return True, "inference.py imports successfully"

def main():
    checks = [
        ("Files", check_files),
        ("Env Vars", check_env_vars),
        ("Import", import_inference),
    ]

    failures = []
    for name, fn in checks:
        try:
            ok, msg = fn()
        except Exception:
            ok = False
            msg = traceback.format_exc()
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {msg}")
        if not ok:
            failures.append(name)

    if failures:
        print(f"\nResult: FAILED checks -> {failures}")
        sys.exit(1)

    print("\nResult: ALL CHECKS PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()
