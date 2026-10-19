# syntax=docker/dockerfile:1

# ---- builder: build the wheel ------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --no-deps -w /wheels .

# ---- runtime: install the wheel with the Claude backend extra ----------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="multi-agent-debate" \
      org.opencontainers.image.description="A swarm of LLM agents that propose, critique, vote and synthesize an answer with confidence." \
      org.opencontainers.image.source="https://github.com/imranmali/multi-agent-debate-system" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    set -eu; \
    wheel="$(ls /wheels/*.whl)"; \
    pip install "${wheel}[anthropic]"

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin debate
USER debate
WORKDIR /home/debate

ENTRYPOINT ["debate"]
CMD ["--help"]
