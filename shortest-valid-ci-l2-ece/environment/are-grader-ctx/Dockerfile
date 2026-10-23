FROM --platform=linux/amd64 public.ecr.aws/docker/library/python:3.13-slim-bookworm@sha256:01f42367a0a94ad4bc17111776fd66e3500c1d87c15bbd6055b7371d39c124fb

ENTRYPOINT []
ENV PYTHONPATH=
ENV PYTHONSAFEPATH=1
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN useradd --create-home --uid 1001 agent

COPY ./requirements.txt /tmp/requirements.txt
COPY ./requirements-verifier.txt /tmp/requirements-verifier.txt
RUN /usr/local/bin/python3 -m pip install --no-cache-dir \
        -r /tmp/requirements.txt -r /tmp/requirements-verifier.txt \
    && rm /tmp/requirements.txt /tmp/requirements-verifier.txt

RUN mkdir -p /app/output /logs/verifier /opt/candrun

COPY . /tests

RUN cp /tests/run_call.py /opt/candrun/run_call.py \
    && chown root:root /opt/candrun /opt/candrun/run_call.py \
    && chmod 755 /opt/candrun \
    && chmod 444 /opt/candrun/run_call.py \
    && chown -R root:root /tests /home/agent \
    && chmod 755 /home/agent \
    && chmod 755 /tmp /var/tmp \
    && chmod 700 /tests/hidden_data /tests/hidden_data/intermediate /tests/hidden_data/final \
    && chmod +x /tests/test.sh /tests/intermediate.sh \
    && find /tests -exec touch -d @1700000000 {} +

WORKDIR /tests

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
