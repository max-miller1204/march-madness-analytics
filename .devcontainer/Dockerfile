FROM ghcr.io/astral-sh/uv:python3.12-trixie
ENV PATH="/root/.local/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends git jq pandoc texlive-xetex texlive-fonts-recommended texlive-plain-generic bubblewrap socat nodejs npm && rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/sandbox-runtime
WORKDIR /workspace
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
