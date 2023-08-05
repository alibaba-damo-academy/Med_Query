#!/usr/bin/env bash
echo "USAGE: sh start_distributed_training.sh train_file_path config_path gpus tag [num_therads_per_process=4] [port=29500]"
echo "the train_file should accept -c -t for config_path and tag respectively"
echo "--local_rank is also needed if torch version lower than 1.10"

TRAIN_FILE=$1
DIST_MODE=$2
CONFIG=$3
GPUS=$4
TAG=$5
NUM_THREADS_PER_PROCESS=$6
PORT=$7

NUM_THREADS_PER_PROCESS=${NUM_THREADS_PER_PROCESS:-4}
PORT=${PORT:-29500}

TMP=${GPUS//,/}
NUM_GPUS=${#TMP}

if [[ $DIST_MODE == "launch" ]]; then
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=$NUM_THREADS_PER_PROCESS \
    python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=$PORT $TRAIN_FILE \
    -c $CONFIG -t $TAG
elif [[ $DIST_MODE == "torchrun" ]]; then
    CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=$NUM_THREADS_PER_PROCESS \
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT $TRAIN_FILE -c $CONFIG -t $TAG
else
    echo "Wrong distributed running mode!"
fi