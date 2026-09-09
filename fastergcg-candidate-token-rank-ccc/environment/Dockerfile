FROM --platform=linux/amd64 gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461

ENV PYTHONHASHSEED=0
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

ENTRYPOINT []

RUN useradd --create-home --uid 1001 agent


RUN /usr/bin/python3 -c "import hashlib, os, shutil, numpy as np, torch; from huggingface_hub import snapshot_download; p = snapshot_download(repo_id='lmsys/vicuna-7b-v1.5', revision='3321f76e3f527bd14065daf69dad9344000a201d', local_dir='/opt/assets/vicuna-7b-v1.5'); sd = torch.load(os.path.join(p, 'pytorch_model-00001-of-00002.bin'), map_location='cpu', weights_only=True); w = sd['model.embed_tokens.weight'].to(torch.float32).numpy(); assert w.shape == (32000, 4096), w.shape; assert hashlib.sha256(w.tobytes()).hexdigest() == '6fae809e95ad8d517dea0a9b3e4389764411168cc6186708af8549e7fa401d03'; np.save('/opt/assets/embedding_matrix.npy', w); os.makedirs('/opt/assets/vicuna-tokenizer', exist_ok=True); [shutil.copy(os.path.join(p, f), '/opt/assets/vicuna-tokenizer/') for f in ('tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json')]; shutil.rmtree(os.path.join(p, '.cache'), ignore_errors=True)"

WORKDIR /app
COPY ./app /app

RUN mkdir -p /app/output /app/work \
    && chown -R agent:agent /app \
    && find /app -exec touch -d @1700000000 {} +

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

RUN git config --system user.email "agent@localhost" && git config --system user.name "agent"

RUN command -v tmux >/dev/null 2>&1 \
    || (apt-get update \
        && apt-get install -y --no-install-recommends tmux \
        && rm -rf /var/lib/apt/lists/*)

USER agent
