#!/bin/bash

docker run --entrypoint ./entrypoint.sh \
        --gpus '"device=0,1"' --shm-size 1g --publish 9123:80 \
        --volume /shd/zzr/models:/models \
        --volume /nvme/zzr/text-generation-inference:/usr/src \
        ghcr.nju.edu.cn/huggingface/text-generation-inference:1.4 \
