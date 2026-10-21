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
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_PROGRESS_BARS=1

RUN useradd --create-home --uid 1001 agent

COPY ./requirements-verifier.txt /tmp/requirements-verifier.txt
RUN /usr/bin/python3 -m pip install --no-cache-dir -r /tmp/requirements-verifier.txt \
    && rm /tmp/requirements-verifier.txt

RUN /usr/bin/python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openai-community/gpt2', revision='607a30d783dfa663caf39e06633721c8d4cfcd7e', local_dir='/opt/assets/gpt2', allow_patterns=['config.json','generation_config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'])" \
    && rm -rf /opt/assets/gpt2/.cache

RUN mkdir -p /app/output /logs/verifier

COPY . /tests

RUN /usr/bin/python3 /tests/prepare_hidden_data.py

RUN chown -R root:root /tests \
    && chmod 700 /tests/hidden_data \
    && chmod +x /tests/test.sh /tests/intermediate.sh \
    && find /tests -exec touch -d @1700000000 {} +

WORKDIR /tests

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
