FROM --platform=linux/amd64 gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461

ENV PYTHONHASHSEED=0
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV OMP_NUM_THREADS=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

ENTRYPOINT []

RUN useradd --create-home --uid 1001 agent

RUN command -v tmux >/dev/null 2>&1 \
    || (apt-get update \
        && apt-get install -y --no-install-recommends tmux \
        && rm -rf /var/lib/apt/lists/*)

RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 /usr/bin/python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='zai-org/CogVideoX-2b', revision='1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01', local_dir='/opt/assets/cogvideox', allow_patterns=['model_index.json', 'transformer/*', 'vae/*', 'scheduler/*'])" \
 && /usr/bin/python3 -c "import hashlib; d=lambda f: hashlib.sha256(open('/opt/assets/cogvideox/'+f,'rb').read()).hexdigest(); assert d('transformer/diffusion_pytorch_model.safetensors')=='8fbb6a5e67c70885a8ed8e33df144ac61253e45977be5035fa18cfdf77d386c7'; assert d('vae/diffusion_pytorch_model.safetensors')=='a410e48d988c8224cef392b68db0654485cfd41f345f4a3a81d3e6b765bb995e'" \
 && rm -rf /opt/assets/cogvideox/.cache

WORKDIR /app
COPY ./app /app

RUN mkdir -p /app/output /app/dev_work \
    && chown -R agent:agent /app \
    && find /app -exec touch -d @1700000000 {} +

RUN git config --system user.email "agent@localhost" \
    && git config --system user.name "agent"

USER agent
