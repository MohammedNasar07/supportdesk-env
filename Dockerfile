FROM python:3.11-slim

# Create a non-root user for security and compliance with some HF Space requirements
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Pre-install requirements for faster builds
COPY --chown=user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application
COPY --chown=user . $HOME/app

# Ensure logs are not buffered
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# Start the unified FastAPI + Gradio server
CMD ["python", "server/app.py"]