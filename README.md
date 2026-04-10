---
title: SupportFlow Arena
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# SupportFlow Arena: OpenEnv Hackathon Project

## Overview
SupportFlow Arena is a deterministic customer support triage environment. An AI agent reads support tickets and outputs structured JSON decisions (category, priority, clarification, escalation, and response).

## Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Environment Variables**:
   - `HF_TOKEN`: Required. Your Hugging Face API token.
   - `API_BASE_URL`: Optional (defaults to OpenAI).
   - `MODEL_NAME`: Optional (defaults to gpt-4.1-mini).

3. **Run Inference**:
   ```bash
   python inference.py
   ```

## Rules & Constraints
- `inference.py` must be in the project root.
- All scores are strictly clamped between **0.01 and 0.99**.
- Logging uses the exact `[START]`, `[STEP]`, and `[END]` format.
- No internal newlines allowed in log outputs.
- Uses only the official OpenAI Python client for LLM calls.

## Project Structure
- `inference.py`: Main entry point for the validator.
- `app.py`: Dual-purpose FastAPI + Gradio server.
- `src/`: Core logic (schemas, policy, env, grader, generator).
- `data/`: Asset storage (tickets.json).
- `demo.py`: CLI verification script.
