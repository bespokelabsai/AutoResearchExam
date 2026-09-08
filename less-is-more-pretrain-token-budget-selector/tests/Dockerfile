FROM --platform=linux/amd64 gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461

ENTRYPOINT []
ENV PYTHONPATH=
ENV PYTHONSAFEPATH=1
ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

RUN useradd --create-home --uid 1001 agent

COPY ./requirements-verifier.txt /tmp/requirements-verifier.txt
RUN /usr/bin/python3 -m pip install --no-cache-dir -r /tmp/requirements-verifier.txt \
    && rm /tmp/requirements-verifier.txt

RUN /usr/bin/python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openai-community/gpt2', revision='607a30d783dfa663caf39e06633721c8d4cfcd7e', local_dir='/opt/assets/gpt2', allow_patterns=['config.json','generation_config.json','merges.txt','vocab.json','tokenizer.json','model.safetensors']); snapshot_download(repo_id='EleutherAI/pythia-160m', revision='50f5173d932e8e61f858120bcb800b97af589f46', local_dir='/opt/assets/pythia-160m', allow_patterns=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors'])" && rm -rf /opt/assets/gpt2/.cache /opt/assets/pythia-160m/.cache

COPY . /tests

RUN set -eux; \
    REV=1588ec454efa1a09f29cd18ddd04fe05fc8653a2; \
    B="https://huggingface.co/datasets/allenai/c4/resolve/$REV"; \
    TK=/opt/assets/gpt2/tokenizer.json; \
    mkdir -p /tmp/corpus /tests/hidden_data; \
    curl -sSL -o /tmp/corpus/valF.json.gz "$B/en/c4-validation.00003-of-00008.json.gz"; \
    curl -sSL -o /tmp/corpus/valI.json.gz "$B/en/c4-validation.00005-of-00008.json.gz"; \
    curl -sSL -o /tmp/corpus/trF0.json.gz "$B/en/c4-train.00777-of-01024.json.gz"; \
    curl -sSL -o /tmp/corpus/trF1.json.gz "$B/en/c4-train.00778-of-01024.json.gz"; \
    curl -sSL -o /tmp/corpus/ncF.json.gz  "$B/en.noclean/c4-train.00777-of-07168.json.gz"; \
    curl -sSL -o /tmp/corpus/trI0.json.gz "$B/en/c4-train.00888-of-01024.json.gz"; \
    curl -sSL -o /tmp/corpus/trI1.json.gz "$B/en/c4-train.00889-of-01024.json.gz"; \
    curl -sSL -o /tmp/corpus/ncI.json.gz  "$B/en.noclean/c4-train.00888-of-07168.json.gz"; \
    export TOKENIZERS_PARALLELISM=true; \
    /usr/bin/python3 /tests/build_data.py eval --shards /tmp/corpus/valF.json.gz \
      --out /tests/hidden_data/final/eval --tokenizer $TK --target-tokens 2000000; \
    /usr/bin/python3 /tests/build_data.py pool --shards /tmp/corpus/trF0.json.gz /tmp/corpus/trF1.json.gz \
      --shards-b /tmp/corpus/ncF.json.gz --target-tokens 250000000 --target-tokens-b 62500000 \
      --out /tests/hidden_data/final/pool --tokenizer $TK \
      --guard /tests/hidden_data/final/eval/guard.json; \
    /usr/bin/python3 /tests/build_data.py eval --shards /tmp/corpus/valI.json.gz \
      --out /tests/hidden_data/intermediate/eval --tokenizer $TK --target-tokens 2000000; \
    /usr/bin/python3 /tests/build_data.py pool --shards /tmp/corpus/trI0.json.gz /tmp/corpus/trI1.json.gz \
      --shards-b /tmp/corpus/ncI.json.gz --target-tokens 250000000 --target-tokens-b 62500000 \
      --out /tests/hidden_data/intermediate/pool --tokenizer $TK \
      --guard /tests/hidden_data/intermediate/eval/guard.json; \
    rm -rf /tmp/corpus

RUN mkdir -p /app/output /logs/verifier

RUN chown -R root:root /tests \
    && chmod 700 /tests/hidden_data \
    && chmod +x /tests/test.sh /tests/intermediate.sh \
    && find /tests -exec touch -d @1700000000 {} +

WORKDIR /tests

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
