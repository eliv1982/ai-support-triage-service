FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Unprivileged user. It owns only /data, where the SQLite file lives; the code in /app
# stays root-owned and read-only to it.
RUN useradd --system --uid 10001 --no-create-home app \
    && mkdir /data \
    && chown app:app /data

COPY app ./app

USER app

# The default DATABASE_URL (sqlite:///./app.db) is relative to the working directory,
# so run from the writable data dir. PYTHONPATH keeps `app.main` importable from there.
WORKDIR /data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
