FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_VIRTUALENVS_IN_PROJECT=false
ENV POETRY_NO_INTERACTION=1
ENV MLFLOW_TRACKING_URI=/app/mlruns

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.4

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-ansi

COPY src/ ./src/
RUN poetry install --only main --no-ansi

RUN mkdir -p data/raw data/processed models reports mlruns

EXPOSE 5000

CMD ["churn-train"]
