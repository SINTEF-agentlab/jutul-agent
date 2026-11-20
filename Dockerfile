# Container for running evals (and headless Makker) in isolation.
#
#   docker compose run --rm eval canary
#
# Julia's depot and the Makker state home are volumes (see
# docker-compose.yml): the first simulator run pays the full env instantiate
# and precompile, every run after reuses the caches. Without the volumes each
# run starts cold, which costs many minutes per simulator suite.

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    xvfb libgl1 libgl1-mesa-dri mesa-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Julia via juliaup, pinned to the same channel CI uses.
RUN curl -fsSL https://install.julialang.org | sh -s -- -y --default-channel 1.12
ENV PATH="/root/.juliaup/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --extra eval --no-install-project
COPY . .
RUN uv sync --locked --extra eval

# Caches live on volumes so they outlive the container.
ENV JULIA_DEPOT_PATH=/depot \
    XDG_DATA_HOME=/state

ENTRYPOINT ["uv", "run", "makker"]
CMD ["eval", "--list"]
