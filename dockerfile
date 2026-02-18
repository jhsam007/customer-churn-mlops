# Base Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml .

RUN pip install --upgrade pip \
    && pip install .

# Copy project files
COPY . .

# Make sure model directory exists
RUN mkdir -p models

# Expose FastAPI port
EXPOSE 8000

# Run API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]