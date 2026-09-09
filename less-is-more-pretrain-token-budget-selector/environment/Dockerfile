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
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

ENTRYPOINT []

RUN useradd --create-home --uid 1001 agent

RUN command -v tmux >/dev/null 2>&1 \
    || (apt-get update \
        && apt-get install -y --no-install-recommends tmux \
        && rm -rf /var/lib/apt/lists/*)

RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 /usr/bin/python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openai-community/gpt2', revision='607a30d783dfa663caf39e06633721c8d4cfcd7e', local_dir='/opt/assets/gpt2', allow_patterns=['config.json','generation_config.json','merges.txt','vocab.json','tokenizer.json','model.safetensors']); snapshot_download(repo_id='EleutherAI/pythia-160m', revision='50f5173d932e8e61f858120bcb800b97af589f46', local_dir='/opt/assets/pythia-160m', allow_patterns=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors'])" && rm -rf /opt/assets/gpt2/.cache /opt/assets/pythia-160m/.cache

COPY ./build_data.py /tmp/build_data.py
RUN set -eux; \
    REV=1588ec454efa1a09f29cd18ddd04fe05fc8653a2; \
    B="https://huggingface.co/datasets/allenai/c4/resolve/$REV"; \
    mkdir -p /tmp/corpus /app/data; \
    curl -sSL -o /tmp/corpus/val0.json.gz  "$B/en/c4-validation.00000-of-00008.json.gz"; \
    curl -sSL -o /tmp/corpus/tr0.json.gz   "$B/en/c4-train.00512-of-01024.json.gz"; \
    curl -sSL -o /tmp/corpus/tr1.json.gz   "$B/en/c4-train.00513-of-01024.json.gz"; \
    curl -sSL -o /tmp/corpus/nc0.json.gz   "$B/en.noclean/c4-train.00512-of-07168.json.gz"; \
    TOKENIZERS_PARALLELISM=true /usr/bin/python3 /tmp/build_data.py eval \
      --shards /tmp/corpus/val0.json.gz --out /app/data/dev_eval \
      --tokenizer /opt/assets/gpt2/tokenizer.json --target-tokens 2000000; \
    TOKENIZERS_PARALLELISM=true /usr/bin/python3 /tmp/build_data.py pool \
      --shards /tmp/corpus/tr0.json.gz /tmp/corpus/tr1.json.gz \
      --shards-b /tmp/corpus/nc0.json.gz \
      --target-tokens 250000000 --target-tokens-b 62500000 \
      --out /app/data/dev_pool --tokenizer /opt/assets/gpt2/tokenizer.json \
      --guard /app/data/dev_eval/guard.json; \
    rm -rf /tmp/corpus /tmp/build_data.py /app/data/dev_pool/guard.json /app/data/dev_eval/guard.json

WORKDIR /app
COPY ./app /app

RUN mkdir -p /app/output /app/scratch \
    && chown -R agent:agent /app \
    && find /app -exec touch -d @1700000000 {} +

RUN git config --system user.email "agent@localhost" \
    && git config --system user.name "agent"

USER agent
