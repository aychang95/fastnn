#!/usr/bin/env bash

set -e

MODEL_REPOSITORY=${MODEL_REPOSITORY?Variable not set}

docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${MODEL_REPOSITORY?Variable not set}:/models nvcr.io/nvidia/tritonserver:20.10-py3 tritonserver \
    --model-repository=/models \
    --log-verbose=1 \
    --model-control-mode=explicit \
    --load-model=fasterrcnn-resnet50 \
    --load-model=bert-large-cased-whole-word-masking-finetuned-squad \
    --load-model=bert-large-cased-whole-word-masking-finetuned-squad-cpu \
    --load-model=mrm8488.bert-base-portuguese-cased-finetuned-squad-v1-pt \
    --load-model=mrm8488.bert-base-portuguese-cased-finetuned-squad-v1-pt-cpu \
    --load-model=deepset.roberta-base-squad2 \
    --load-model=deepset.roberta-base-squad2-cpu \
    --load-model=deepset.bert-large-uncased-whole-word-masking-squad2 \
    --load-model=deepset.bert-large-uncased-whole-word-masking-squad2-cpu \
    --load-model=fasterrcnn-resnet50-cpu \
    --load-model=distilbert-squad \
    --load-model=distilbert-squad-cpu
