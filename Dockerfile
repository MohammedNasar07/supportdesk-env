FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy every flat file
COPY --chown=user:user models.py       .
COPY --chown=user:user ticket_data.py  .
COPY --chown=user:user graders.py      .
COPY --chown=user:user tasks.py        .
COPY --chown=user:user environment.py  .
COPY --chown=user:user app.py          .
COPY --chown=user:user openenv.yaml    .

USER user
EXPOSE 7860
ENV PORT=7860
CMD ["python", "app.py"]
