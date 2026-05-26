FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/.venv

# Install uv, create virtual environment, and activate it
RUN pip install uv
RUN uv venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Copy dependency definitions
COPY pyproject.toml uv.lock /

# Install dependencies
RUN uv sync --no-cache-dir

# build the runtime image from the builder.
FROM python:3.11-slim AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && apt-get purge -y --auto-remove \
 && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/.venv
ENV PATH="/.venv/bin:$PATH"

# Set user and group
ARG user=appuser
ARG group=appuser
ARG uid=1000
ARG gid=1000
RUN groupadd -g ${gid} ${group}
RUN useradd -u ${uid} -g ${group} -s /bin/sh -m ${user}

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY ./src /workspace/src
ENV PYTHONPATH="/workspace/src:$PYTHONPATH"
COPY ./app /workspace/app
COPY ./logs /workspace/logs
COPY pyproject.toml /workspace
RUN chown -R "${user}":"${group}" /workspace

ENV HF_HOME=/workspace/cache/huggingface
RUN mkdir /workspace/cache
RUN mkdir /workspace/cache/huggingface
RUN mkdir /workspace/cache/dspy_cache
RUN chown -R "${user}":"${group}" /workspace/cache
RUN chmod -R 777 /workspace/cache
# Switch to user and work dir
WORKDIR /workspace
USER ${uid}
ENTRYPOINT ["uvicorn", "app.server:app", "--host", "0.0.0.0"]
