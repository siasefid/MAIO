FROM python:3.11-slim

# Avoid .pyc files and enable clean logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy dependencies and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and artifacts
COPY src ./src
COPY artifacts ./artifacts

# Set PYTHONPATH so imports work
ENV PYTHONPATH=/app/src

# Default model env vars
ENV MODEL_PATH=artifacts/model_baseline.pkl
ENV MODEL_VERSION=v0.1

EXPOSE 8000

CMD ["uvicorn", "risk_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
