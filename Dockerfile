# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1

WORKDIR /build

RUN pip install --no-cache-dir poetry==2.1.3 poetry-plugin-export==1.9.0

COPY pyproject.toml poetry.lock ./
RUN poetry export --only main --without-hashes --format requirements.txt \
      --output requirements.txt \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Bind-mounted from the builder rather than copied: a COPY layer would keep
# every wheel in the image even after a later `rm`.
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip install --no-cache-dir /wheels/* \
 && useradd --create-home --uid 10001 appuser

COPY embedding_provider.py rate_limit.py ./
COPY recommendation_engine ./recommendation_engine
# PYTHONDONTWRITEBYTECODE stops the runtime caching bytecode, so compile it once
# here instead of on every cold start.
RUN python -m compileall -q embedding_provider.py rate_limit.py recommendation_engine

USER appuser

EXPOSE 8080

CMD ["python", "-m", "recommendation_engine.server"]
