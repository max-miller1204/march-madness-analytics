FROM ghcr.io/astral-sh/uv:python3.12-trixie

WORKDIR /workspace
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
