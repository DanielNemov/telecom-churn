FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.4

COPY pyproject.toml poetry.lock poetry.toml ./
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi --no-root

COPY src/ ./src/
RUN poetry install --only main --no-interaction --no-ansi

RUN mkdir -p data/raw data/processed models reports mlruns

ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=/app/mlruns
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 5000

CMD ["churn-train"]
