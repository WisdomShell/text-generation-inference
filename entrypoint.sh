#!/bin/bash

# docker run --gpus '"device=0,1"' --shm-size 1g --publish 9123:80 \
#         --volume /shd/zzr/models:/models \
#         --volume /nvme/zzr/text-generation-inference:/usr/src/server \
#         ghcr.nju.edu.cn/huggingface/text-generation-inference:1.4 \
#         --model-id /models/codeshell-7b \
#         --num-shard 2 \
#         --rope-scaling dynamic \
#         --rope-factor 8 \
#         --max-input-length 31000 \
#         --max-total-tokens 32768 \
#         --max-batch-prefill-tokens 31000 \
#         --max-stop-sequences 12
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# source "$HOME/.cargo/env"
# cd router && cargo install --path .
make clean
# BUILD_EXTENSIONS=True make install-server
make run-llama2
# make run-codeshell