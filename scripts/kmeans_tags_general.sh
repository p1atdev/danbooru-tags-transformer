#!/bin/bash

MODEL_NAME="p1atdev/dart-v3-vectors-opt_17-shuffled"
N_CLUSTERS=1024
N_INIT=10
MAX_ITER=300
OUTPUT_PATH="data/general_1024cluster_6.json"
TAGS="data/general_tags.txt"

python ./train/cluster/kmeans_tags.py \
    --model_name $MODEL_NAME \
    --n_clusters $N_CLUSTERS \
    --n_init $N_INIT \
    --max_iter $MAX_ITER \
    --output_path $OUTPUT_PATH \
    --tags $TAGS
