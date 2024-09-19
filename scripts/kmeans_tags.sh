#!/bin/bash

MODEL_NAME="p1atdev/dart-v3-vectors-opt_7-shuffled"
N_CLUSTERS=1440
N_INIT=25
MAX_ITER=1000
OUTPUT_PATH="data/cluster_map_1440c2.json"

python ./train/cluster/kmeans_tags.py \
    --model_name $MODEL_NAME \
    --n_clusters $N_CLUSTERS \
    --n_init $N_INIT \
    --max_iter $MAX_ITER \
    --output_path $OUTPUT_PATH
