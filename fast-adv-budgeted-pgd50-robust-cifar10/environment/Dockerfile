FROM --platform=linux/amd64 gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461 AS data

ARG CIFAR_REVISION=0b2714987fa478483af9968de7c934580d0bb9a2
ENTRYPOINT []
WORKDIR /build
COPY ./materialize_cifar.py /build/materialize_cifar.py
RUN curl -fsSL --retry 8 --retry-all-errors --retry-delay 5 --max-time 3600 \
        -o /build/train.parquet \
        "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/${CIFAR_REVISION}/plain_text/train-00000-of-00001.parquet" \
    && mkdir -p /data \
    && /usr/bin/python3 /build/materialize_cifar.py --split train \
        --parquet /build/train.parquet --out /data/cifar10_train.npz \
    && rm -f /build/train.parquet


FROM --platform=linux/amd64 gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461

ENTRYPOINT []
ENV PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN useradd --create-home --uid 1001 agent

RUN command -v tmux >/dev/null 2>&1 \
    || (apt-get update \
        && apt-get install -y --no-install-recommends tmux \
        && rm -rf /var/lib/apt/lists/*)

WORKDIR /app
COPY ./app /app
COPY --from=data /data/cifar10_train.npz /app/data/cifar10_train.npz

RUN mkdir -p /app/output \
    && chown -R agent:agent /app \
    && find /app -exec touch -d @1700000000 {} +

RUN git config --system user.email "agent@localhost" \
    && git config --system user.name "agent"

USER agent
