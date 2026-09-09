FROM --platform=linux/amd64 gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461

ENTRYPOINT []
ENV PYTHONPATH=
ENV PYTHONSAFEPATH=1
ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1

RUN useradd --create-home --uid 1001 agent

COPY ./requirements-verifier.txt /tmp/requirements-verifier.txt
RUN /usr/bin/python3 -m pip install --no-cache-dir -r /tmp/requirements-verifier.txt \
    && rm /tmp/requirements-verifier.txt

RUN /usr/bin/python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='zai-org/CogVideoX-2b', revision='1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01', local_dir='/opt/assets/cogvideox', allow_patterns=['model_index.json', 'transformer/*', 'vae/*', 'scheduler/*'])" \
 && /usr/bin/python3 -c "import hashlib; d=lambda f: hashlib.sha256(open('/opt/assets/cogvideox/'+f,'rb').read()).hexdigest(); assert d('transformer/diffusion_pytorch_model.safetensors')=='8fbb6a5e67c70885a8ed8e33df144ac61253e45977be5035fa18cfdf77d386c7'; assert d('vae/diffusion_pytorch_model.safetensors')=='a410e48d988c8224cef392b68db0654485cfd41f345f4a3a81d3e6b765bb995e'" \
 && rm -rf /opt/assets/cogvideox/.cache

RUN chmod 700 /opt/assets/cogvideox

RUN mkdir -p /app/output /logs/verifier

COPY . /tests

RUN chown -R root:root /tests \
    && chmod 700 /tests/hidden_data \
    && chmod +x /tests/test.sh /tests/intermediate.sh \
    && find /tests -exec touch -d @1700000000 {} +

WORKDIR /tests

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
