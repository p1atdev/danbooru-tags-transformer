#!/bin/bash

MODEL_NAME="p1atdev/dart-v3-vectors-opt_7-shuffled"
N_CLUSTERS=1152
N_INIT=10
MAX_ITER=250
OUTPUT_PATH="data/cluster_map_1152c1.json"
GENERAL_TAGS="data/general_tags.txt"

python ./train/cluster/kmeans_tags.py \
    --model_name $MODEL_NAME \
    --n_clusters $N_CLUSTERS \
    --n_init $N_INIT \
    --max_iter $MAX_ITER \
    --output_path $OUTPUT_PATH \
    --general_tags $GENERAL_TAGS
