FROM --platform=linux/amd64 public.ecr.aws/docker/library/python:3.13-slim-bookworm@sha256:01f42367a0a94ad4bc17111776fd66e3500c1d87c15bbd6055b7371d39c124fb

RUN set -eu; \
    if [ ! -s /etc/ssl/certs/ca-certificates.crt ]; then \
      apt-get update && apt-get install -y --no-install-recommends ca-certificates=20230311+deb12u1; \
    fi; \
    rm -f /tmp/hz-prereq.list; \
    echo 'deb [signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] https://snapshot.debian.org/archive/debian/20260801T000000Z bookworm main' >> /tmp/hz-prereq.list; \
    echo 'deb [signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] https://snapshot.debian.org/archive/debian-security/20260801T000000Z bookworm-security main' >> /tmp/hz-prereq.list; \
    apt-get -o Dir::Etc::sourcelist=/tmp/hz-prereq.list -o Dir::Etc::sourceparts=/dev/null -o Acquire::Check-Valid-Until=false update; \
    apt-get -o Dir::Etc::sourcelist=/tmp/hz-prereq.list -o Dir::Etc::sourceparts=/dev/null -o Acquire::Check-Valid-Until=false install -y --no-install-recommends tmux=3.3a-3 asciinema=2.2.0-1 git=1:2.39.5-0+deb12u3; \
    rm -f /tmp/hz-prereq.list; rm -rf /var/lib/apt/lists/*


ENTRYPOINT []
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8

RUN useradd --create-home --uid 1001 agent

COPY ./requirements.txt /tmp/requirements.txt
RUN /usr/local/bin/python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

RUN /usr/local/bin/python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openai-community/gpt2', revision='607a30d783dfa663caf39e06633721c8d4cfcd7e', allow_patterns=['model.safetensors', 'config.json'], local_dir='/opt/model')" \
    && rm -rf /opt/model/.cache \
    && find /opt/model -type d -exec chmod 755 {} + \
    && find /opt/model -type f -exec chmod 644 {} +

WORKDIR /app
COPY ./app /app

RUN mkdir -p /app/output \
    && chown -R agent:agent /app \
    && find /app -exec touch -d @1700000000 {} +

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

RUN git config --system user.email "agent@localhost" \
    && git config --system user.name "autoresearch agent"
USER agent
