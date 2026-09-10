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

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
 && rm -rf /wheels \
 && useradd --create-home --uid 10001 appuser

COPY embedding_provider.py rate_limit.py ./
COPY recommendation_engine ./recommendation_engine

USER appuser

EXPOSE 8080

CMD ["python", "-m", "recommendation_engine.server"]
