FROM ghcr.io/astral-sh/uv:python3.12-trixie
ENV PATH="/root/.local/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends git jq && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
