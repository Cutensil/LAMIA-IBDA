#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
PRE_INPUT=$3
POST_INPUT=$4
GPUS=$5
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Keep the output timestamp identical in every process so that all ranks write
# into the same default output directory.
export LAMIA_INFER_TIMESTAMP=${LAMIA_INFER_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}

export PYTHONPATH="$(dirname "$0")/..${PYTHONPATH:+:$PYTHONPATH}"
# Use the torchrun entry point through the active Python interpreter.
exec python -m torch.distributed.run \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    "$(dirname "$0")/infer.py" \
    "$CONFIG" \
    "$CHECKPOINT" \
    "$PRE_INPUT" \
    "$POST_INPUT" \
    "${@:6}"
