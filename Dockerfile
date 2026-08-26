FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=300 -r requirements.txt

COPY src ./src
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]