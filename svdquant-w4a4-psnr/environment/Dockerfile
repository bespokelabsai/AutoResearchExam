FROM --platform=linux/amd64 gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461
ENV PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 OMP_NUM_THREADS=1

ENTRYPOINT []

RUN useradd --create-home --uid 1001 agent
COPY ./requirements.txt /tmp/requirements.txt
RUN /usr/bin/python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
RUN /usr/bin/python3 -c "from huggingface_hub import snapshot_download; snapshot_download('PixArt-alpha/PixArt-Sigma-XL-2-1024-MS', revision='e102b3591cc82e97071b8b4cb90d834d0c487207', local_dir='/opt/pixart-sigma', ignore_patterns=['asset/*', 'README.md', '.gitattributes'])" \
    && rm -rf /root/.cache/huggingface

WORKDIR /app
COPY ./app/data/calibration_prompts.jsonl /app/data/calibration_prompts.jsonl
COPY ./entrypoint.sh /usr/local/bin/task-entrypoint
RUN chown -R agent:agent /app \
    && chmod +x /usr/local/bin/task-entrypoint \
    && find /app -exec touch -d @1700000000 {} +
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

RUN command -v git >/dev/null 2>&1 \
    || (apt-get update \
        && apt-get install -y --no-install-recommends git \
        && rm -rf /var/lib/apt/lists/*)
RUN git config --system user.email "agent@localhost" \
    && git config --system user.name "agent"

RUN command -v tmux >/dev/null 2>&1 \
    || (apt-get update \
        && apt-get install -y --no-install-recommends tmux \
        && rm -rf /var/lib/apt/lists/*)
USER agent
ENTRYPOINT ["/usr/local/bin/task-entrypoint"]
