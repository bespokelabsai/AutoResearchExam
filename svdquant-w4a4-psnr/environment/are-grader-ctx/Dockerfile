FROM --platform=linux/amd64 gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461

ENTRYPOINT []
ENV PYTHONPATH=
ENV PYTHONSAFEPATH=1
ENV PYTHONNOUSERSITE=1
ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

RUN useradd --create-home --uid 1001 agent
COPY ./requirements.txt /tmp/requirements.txt
RUN /usr/bin/python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
RUN /usr/bin/python3 -c "from huggingface_hub import snapshot_download; snapshot_download('PixArt-alpha/PixArt-Sigma-XL-2-1024-MS', revision='e102b3591cc82e97071b8b4cb90d834d0c487207', local_dir='/opt/pixart-sigma', ignore_patterns=['asset/*', 'README.md', '.gitattributes'])" \
    && rm -rf /root/.cache/huggingface

COPY ./requirements-verifier.txt /tmp/requirements-verifier.txt
RUN /usr/bin/python3 -m pip install --no-cache-dir -r /tmp/requirements-verifier.txt \
    && rm /tmp/requirements-verifier.txt
RUN mkdir -p /app/output /logs/verifier /tests/hidden_data/intermediate /tests/hidden_data/final
COPY ./test.sh /tests/test.sh
COPY ./intermediate.sh /tests/intermediate.sh
COPY ./test_state.py /tests/test_state.py
COPY ./compute_reward.py /tests/compute_reward.py
COPY ./grader_core.py /tests/grader_core.py
COPY ./candidate_runner.py /tests/candidate_runner.py
COPY ./hidden_data/intermediate/prompts.jsonl /tests/hidden_data/intermediate/prompts.jsonl
COPY ./hidden_data/final/prompts.jsonl /tests/hidden_data/final/prompts.jsonl
COPY ./app_data/calibration_prompts.jsonl /app/data/calibration_prompts.jsonl
RUN chown -R root:root /tests /logs/verifier /opt/pixart-sigma \
    && chmod 700 /tests /tests/hidden_data /tests/hidden_data/intermediate /tests/hidden_data/final \
    && chmod +x /tests/test.sh /tests/intermediate.sh \
    && find /tests /app/data -exec touch -d @1700000000 {} +
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
WORKDIR /tests
