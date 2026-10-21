FROM --platform=linux/amd64 gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461

ENTRYPOINT []
ENV PYTHONPATH=
ENV PYTHONSAFEPATH=1
ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN useradd --create-home --uid 1001 agent

COPY ./requirements-verifier.txt /tmp/requirements-verifier.txt
RUN /usr/bin/python3 -m pip install --no-cache-dir -r /tmp/requirements-verifier.txt \
    && rm /tmp/requirements-verifier.txt

RUN /usr/bin/python3 -c "import hashlib, os, shutil, numpy as np, torch; from huggingface_hub import snapshot_download; p = snapshot_download(repo_id='lmsys/vicuna-7b-v1.5', revision='3321f76e3f527bd14065daf69dad9344000a201d', local_dir='/opt/assets/vicuna-src', allow_patterns=['pytorch_model-00001-of-00002.bin', 'pytorch_model.bin.index.json', 'config.json', 'tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json']); sd = torch.load(os.path.join(p, 'pytorch_model-00001-of-00002.bin'), map_location='cpu', weights_only=True); w = sd['model.embed_tokens.weight'].to(torch.float32).numpy(); assert w.shape == (32000, 4096), w.shape; assert hashlib.sha256(w.tobytes()).hexdigest() == '6fae809e95ad8d517dea0a9b3e4389764411168cc6186708af8549e7fa401d03'; np.save('/opt/assets/embedding_matrix.npy', w); os.makedirs('/opt/assets/vicuna-tokenizer', exist_ok=True); [shutil.copy(os.path.join(p, f), '/opt/assets/vicuna-tokenizer/') for f in ('tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json')]; del sd, w; shutil.rmtree(p, ignore_errors=True)"

RUN mkdir -p /app/output /logs/verifier

COPY . /tests

RUN chown -R root:root /tests \
    && chmod 700 /tests/hidden_data \
    && chmod +x /tests/test.sh /tests/intermediate.sh \
    && find /tests -exec touch -d @1700000000 {} +

WORKDIR /tests

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
