FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir mlflow

ENV PORT=8080
EXPOSE 8080

# Run MLflow in foreground as PID 1
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "8080", "--allowed-hosts", "*", "--cors-allowed-origins", "*"]
