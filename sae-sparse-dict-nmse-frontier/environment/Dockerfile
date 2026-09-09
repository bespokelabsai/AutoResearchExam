FROM --platform=linux/amd64 gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461
ENV PYTHONHASHSEED=0
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

ENTRYPOINT []

RUN useradd --create-home --uid 1001 agent

RUN command -v tmux >/dev/null 2>&1 \
    || (apt-get update \
        && apt-get install -y --no-install-recommends tmux \
        && rm -rf /var/lib/apt/lists/*)


RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 /usr/bin/python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openai-community/gpt2', revision='607a30d783dfa663caf39e06633721c8d4cfcd7e', local_dir='/opt/assets/gpt2', allow_patterns=['config.json','generation_config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'])" \
    && rm -rf /opt/assets/gpt2/.cache

RUN mkdir -p /app/data/openwebtext \
    && HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 /usr/bin/python3 -c "from huggingface_hub import hf_hub_download; [hf_hub_download(repo_id='Skylion007/openwebtext', repo_type='dataset', revision='79d93d786212f7344586290adb811d4ae6a1762c', filename=f, local_dir='/app/data/openwebtext') for f in ['plain_text/train-00000-of-00080.parquet', 'plain_text/train-00001-of-00080.parquet']]" \
    && rm -rf /app/data/openwebtext/.cache \
    && chown -R agent:agent /app/data

WORKDIR /app
COPY ./app /app

RUN mkdir -p /app/output /app/work \
    && chown agent:agent /app \
    && chown -R agent:agent /app/output /app/work /app/extract_acts.py \
    && chmod 0755 /app/extract_acts.py \
    && find /app/output /app/work /app/extract_acts.py -exec touch -d @1700000000 {} +

RUN git config --system user.email "agent@localhost" \
    && git config --system user.name "agent"

USER agent
