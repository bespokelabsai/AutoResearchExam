FROM --platform=linux/amd64 public.ecr.aws/docker/library/python:3.13-slim-bookworm@sha256:01f42367a0a94ad4bc17111776fd66e3500c1d87c15bbd6055b7371d39c124fb

ENTRYPOINT []

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgomp1=12.2.0-14+deb12u1 \
 && rm -rf /var/lib/apt/lists/* \
 && command -v setsid \
 && command -v runuser

RUN useradd --create-home --uid 1001 agent

COPY requirements.txt /tmp/requirements.txt
COPY requirements-verifier.txt /tmp/requirements-verifier.txt
RUN python3 -m pip install --no-cache-dir \
      -r /tmp/requirements.txt \
      -r /tmp/requirements-verifier.txt \
 && rm -f /tmp/requirements.txt /tmp/requirements-verifier.txt

COPY . /tests/
RUN mkdir -p /app/output /logs/verifier /grade_stage /grade_work \
 && chmod +x /tests/test.sh /tests/intermediate.sh \
 && chown -R root:root /tests \
 && chmod -R go-rwx /tests \
 && chmod 700 /tests/hidden_data /tests/hidden_data/intermediate /tests/hidden_data/final /tests/hidden_data/shared

WORKDIR /tests

ENV PYTHONPATH=
ENV PYTHONSAFEPATH=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

ENV PYTHONHASHSEED=0
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV OPENBLAS_NUM_THREADS=8
