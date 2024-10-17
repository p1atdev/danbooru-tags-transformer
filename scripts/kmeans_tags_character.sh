#!/bin/bash

MODEL_NAME="p1atdev/dart-v3-vectors-opt_14-shuffled"
N_CLUSTERS=2048
N_INIT=10
MAX_ITER=250
OUTPUT_PATH="data/character_2048cluster_1.json"
TAGS="data/character_tags.txt"

python ./train/cluster/kmeans_tags.py \
    --model_name $MODEL_NAME \
    --n_clusters $N_CLUSTERS \
    --n_init $N_INIT \
    --max_iter $MAX_ITER \
    --output_path $OUTPUT_PATH \
    --tags $TAGS
