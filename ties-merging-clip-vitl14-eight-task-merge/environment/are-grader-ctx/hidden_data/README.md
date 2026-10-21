# hidden_data — sealed, 0700 root

Empty in the build context on purpose. `tests/Dockerfile` runs `build_assets.py --role verifier`
above `COPY . /tests`, which materializes

    hidden_data/<panel>/<task>/labels.npy        int64 class ids, row-aligned with the panel images
    hidden_data/<panel>/<task>/classes.json      the dataset's ClassLabel names, head row order
    hidden_data/<panel>/<task>/split_audit.json  row counts and the dedup audit asserted at grade time

for `panel` in {intermediate, final} and the eight tasks. The panels' IMAGES live outside this
tree, under /opt/eval, because merge() is handed them unlabeled; only the labels are sealed.
A later layer chmods this directory to 0700 root, so the uid-1001 candidate cannot read it.
