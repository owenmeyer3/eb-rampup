FROM python:3.11-slim

# XGBoost uses OpenMP for multithreading
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[pipeline, api]"

# Cloud Run CMD sends requests to $PORT. Pipeline and batch jobs override this command.
# ${PORT:-8080} means "use the PORT environment variable, or 8080 if it isn't set." Cloud Run sets PORT
CMD ["sh", "-c", "exec uvicorn eb_rampup.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
