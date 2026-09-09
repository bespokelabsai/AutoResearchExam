FROM --platform=linux/amd64 public.ecr.aws/docker/library/python:3.13-slim-bookworm@sha256:01f42367a0a94ad4bc17111776fd66e3500c1d87c15bbd6055b7371d39c124fb

ENTRYPOINT []

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git=1:2.39.5-0+deb12u3 \
      libgomp1=12.2.0-14+deb12u1 \
      tmux=3.3a-3 \
      asciinema=2.2.0-1 \
 && rm -rf /var/lib/apt/lists/*

RUN git config --system user.email "agent@localhost" \
 && git config --system user.name "autoresearch agent"

RUN useradd --create-home --uid 1001 agent

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm -f /tmp/requirements.txt

WORKDIR /app
COPY app/ /app/
RUN mkdir -p /app/output \
 && chmod +x /app/local_panel.py \
 && chown -R agent:agent /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV OPENBLAS_NUM_THREADS=8
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

USER agent
