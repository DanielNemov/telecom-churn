FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

RUN mkdir -p data/raw data/processed models reports mlruns

ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=/app/mlruns

EXPOSE 5000

CMD ["python", "src/model/train.py"]
