# Simple CPU-only image
FROM python:3.11-slim

WORKDIR /app

# System deps for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app/app.py
ENV PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["gunicorn", "app.app:app", "--bind", ":8080", "--workers", "1", "--threads", "2", "--timeout", "120"]

