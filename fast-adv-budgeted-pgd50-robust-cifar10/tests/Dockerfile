FROM --platform=linux/amd64 gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461 AS data

ARG CIFAR_REVISION=0b2714987fa478483af9968de7c934580d0bb9a2
ENTRYPOINT []
WORKDIR /build
COPY ./materialize_cifar.py /build/materialize_cifar.py
RUN for split in train test; do \
        curl -fsSL --retry 8 --retry-all-errors --retry-delay 5 --max-time 3600 \
            -o "/build/${split}.parquet" \
            "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/${CIFAR_REVISION}/plain_text/${split}-00000-of-00001.parquet" \
        || exit 1; \
    done \
    && mkdir -p /data \
    && /usr/bin/python3 /build/materialize_cifar.py --split train \
        --parquet /build/train.parquet --out /data/cifar10_train.npz \
    && /usr/bin/python3 /build/materialize_cifar.py --split test \
        --parquet /build/test.parquet --out /data/cifar10_test.npz \
    && rm -f /build/train.parquet /build/test.parquet


FROM --platform=linux/amd64 gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461

ENTRYPOINT []
ENV PYTHONPATH=
ENV PYTHONSAFEPATH=1
ENV PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN useradd --create-home --uid 1001 agent

RUN mkdir -p /app/output /app/data /logs/verifier

COPY ./preact_resnet.py /app/preact_resnet.py
COPY --from=data /data/cifar10_train.npz /app/data/cifar10_train.npz

COPY . /tests
COPY --from=data /data/cifar10_test.npz /tests/hidden_data/cifar10_test.npz

RUN chown -R root:root /tests /app/preact_resnet.py /app/data \
    && chmod 700 /tests/hidden_data \
    && chmod +x /tests/test.sh /tests/intermediate.sh \
    && chmod 444 /app/preact_resnet.py /app/data/cifar10_train.npz \
    && find /tests -exec touch -d @1700000000 {} +

WORKDIR /tests
