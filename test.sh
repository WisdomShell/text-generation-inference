#!/bin/bash

# docker run --entrypoint ./entrypoint.sh \
#         --gpus '"device=0,1"' --shm-size 1g --publish 9123:80 \
#         --volume /shd/zzr/models:/models \
#         --volume /nvme/zzr/text-generation-inference:/usr/src \
#         ghcr.nju.edu.cn/huggingface/text-generation-inference:1.4 \


docker run --gpus '"device=0,1"' --shm-size 1g --publish 9123:80 \
        --volume /nvme/zzr/text-generation-inference/server/text_generation_server:/opt/conda/lib/python3.10/site-packages/text_generation_server \
        --volume /shd/zzr/models:/models \
        ghcr.nju.edu.cn/huggingface/text-generation-inference:1.4 \
        --model-id /models/codellama-7b \
        --num-shard 2 \
        --rope-scaling dynamic \
        --rope-factor 8 \
        --max-input-length 31000 \
        --max-total-tokens 32768 \
        --max-batch-prefill-tokens 31000 \
        --max-stop-sequences 12