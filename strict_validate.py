import subprocess
import sys
import re

START_RE = re.compile(r'^\[START\] task=(.+?) env=(.+?) model=(.+?)$')
STEP_RE = re.compile(
    r'^\[STEP\] step=(\d+) action=(.+?) reward=(-?\d+\.\d{2}) done=(true|false) error=(.+?)$'
)
END_RE = re.compile(r'^\[END\] success=(true|false) steps=(\d+) rewards=(.*)$')

def run_inference():
    # We use the existing environment (HF_TOKEN should be set by user)
    env = os.environ.copy()
        
    result = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True,
        text=True,
        env=env
    )
    return result.returncode, result.stdout.strip().splitlines(), result.stderr

def validate(lines):
    if not lines:
        return False, "No stdout produced"

    # Support multiple START/STEP/END blocks (one per task)
    # The validator usually checks blocks.
    
    # Let's find all blocks
    blocks = []
    current_block = []
    for line in lines:
        if line.startswith("[START]"):
            if current_block: blocks.append(current_block)
            current_block = [line]
        elif line.startswith("[STEP]") or line.startswith("[END]"):
            current_block.append(line)
    if current_block: blocks.append(current_block)

    if not blocks:
        return False, "No [START]...[END] blocks found"

    for i, block in enumerate(blocks):
        start_line = block[0]
        end_lines = [l for l in block if l.startswith("[END]")]
        step_lines = [l for l in block if l.startswith("[STEP]")]

        if not START_RE.match(start_line):
            return False, f"Block {i}: Invalid [START] format: {start_line}"

        if len(end_lines) != 1:
            return False, f"Block {i}: Expected exactly 1 [END], found {len(end_lines)}"
        
        end_line = end_lines[0]
        end_match = END_RE.match(end_line)
        if not end_match:
            return False, f"Block {i}: Invalid [END] format: {end_line}"

        for line in step_lines:
            m = STEP_RE.match(line)
            if not m:
                return False, f"Block {i}: Invalid [STEP] format: {line}"

            step_num = int(m.group(1))
            reward = m.group(3)
            # done = m.group(4)
            # error = m.group(5)

            if step_num < 1:
                return False, f"Block {i}: Step number must be >= 1: {line}"
            if not re.match(r'^-?\d+\.\d{2}$', reward):
                return False, f"Block {i}: Reward must have exactly 2 decimals: {line}"

        end_steps = int(end_match.group(2))
        if end_steps != len(step_lines):
            return False, f"Block {i}: [END] steps={end_steps} but found {len(step_lines)} [STEP] lines"

    return True, f"Found {len(blocks)} valid [START]...[END] blocks"

def main():
    code, lines, stderr = run_inference()

    print("=== STDOUT ===")
    for line in lines:
        print(line)

    print("\n=== STDERR ===")
    print(stderr)

    ok, msg = validate(lines)
    if not ok:
        print(f"\nFAIL: {msg}")
        sys.exit(1)

    print(f"\nPASS: {msg}")
    sys.exit(0)

if __name__ == "__main__":
    import os
    main()
