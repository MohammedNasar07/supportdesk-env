# 🏆 SupportFlow Arena: OpenEnv Winner's Edition

SupportFlow Arena is a professional reinforcement learning environment designed to train and benchmark AI customer support agents. It simulates a high-stakes support desk where agents must categorize tickets, prioritize urgency, detect security risks, and draft policy-compliant responses.

## 🌟 Why SupportFlow Arena?
- **Real-World Impact**: Directly maps to common business workflows in SaaS support.
- **Deterministic Grading**: High-precision scoring (0.0–1.0) for every agent decision.
- **Multi-Step Reasoning**: Goes beyond simple classification to include policy enforcement and clarification logic.
- **Winner-Ready Architecture**: Follows best practices for modular, scalable AI environments.

## 📂 Project Structure
```text
repo/
├── inference.py     # Main evaluation entry point
├── app.py           # Gradio demo for Hugging Face Spaces
├── Dockerfile       # Containerized runtime configuration
├── README.md        # This storyteller guide
│
├── src/             # Modular source code
│   ├── env/         # SupportDesk environment and schemas
│   ├── graders/     # Task-specific grading logic
│   └── utils/       # Shared formatting and validation
│
├── configs/         # Task and Prompt configurations (YAML)
└── assets/          # Professional support ticket dataset (JSON)
```

## 🚀 Tasks & Grading
SuppportFlow Arena validates agents across 3 specialized tasks:
1. **Ticket Categorization**: Perfect mapping to Billing, Tech, or Account issues.
2. **Support Triage**: Comprehensive category, priority, and routing analysis.
3. **Full Resolution**: Multi-constraint response drafting following empathy and policy rules.

## 🛠️ Usage
### Local Performance Test
```bash
pip install -r requirements.txt
python demo.py
```

### Live Demo
Run `python app.py` to launch the Gradio UI locally or visit our [Hugging Face Space](https://huggingface.co/spaces/Nasar7/supportdesk-env).

## 📄 Notes
- **Offline First**: No external APIs or cloud databases are required at runtime.
- **Standardized Schema**: Fully compliant with the OpenEnv specification for the Scaler Hackathon.
- **Safe Ranges**: Rewards and scores strictly normalized to the [0.0, 1.0] range.
