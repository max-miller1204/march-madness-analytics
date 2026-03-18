FROM ghcr.io/astral-sh/uv:python3.12-trixie

# Install dependencies needed for Claude Code
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
