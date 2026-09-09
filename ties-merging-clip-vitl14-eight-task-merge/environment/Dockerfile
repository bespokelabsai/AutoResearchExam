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
ENV TQDM_DISABLE=1
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

ENTRYPOINT []

RUN useradd --create-home --uid 1001 agent

RUN command -v tmux >/dev/null 2>&1 \
    || (apt-get update \
        && apt-get install -y --no-install-recommends tmux \
        && rm -rf /var/lib/apt/lists/*)


COPY ./build_assets.py /opt/build_assets.py
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
        /usr/bin/python3 /opt/build_assets.py --role agent --assets /opt/assets --out /app \
        --dataset-cache /tmp/ds_cache --stride 1 \
    && rm -rf /tmp/ds_cache /opt/assets/pretrained/.cache /opt/assets/sun397/.cache \
    && rm -rf /opt/assets/cars/.cache /opt/assets/resisc45/.cache /opt/assets/eurosat/.cache \
    && rm -rf /opt/assets/svhn/.cache /opt/assets/gtsrb/.cache /opt/assets/mnist/.cache \
    && rm -rf /opt/assets/dtd/.cache

WORKDIR /app
COPY ./app /app

RUN mkdir -p /app/output \
    && chown agent:agent /app /app/output /app/eval_dev.py \
    && touch -d @1700000000 /app /app/output /app/eval_dev.py

RUN git config --system user.email "agent@localhost" \
    && git config --system user.name "agent"

USER agent
