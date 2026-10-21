Sealed evaluation material.

This directory is empty in the task source tree and is populated during the verifier image
build by `tests/prepare_hidden_data.py`, which writes `intermediate/` and `final/` -- two
document-disjoint halves of an OpenWebText shard that is not present in the agent image.
It is mode 0700 root-owned inside the verifier and is never copied into the agent image.
