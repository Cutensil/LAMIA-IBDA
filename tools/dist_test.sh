#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Match dist_train.sh: avoid synchronous huge-page compaction for large
# NumPy mask buffers. Set before Python imports; workers inherit the setting.
export NUMPY_MADVISE_HUGEPAGE=${NUMPY_MADVISE_HUGEPAGE:-0}

export PYTHONPATH="$(dirname "$0")/..${PYTHONPATH:+:$PYTHONPATH}"
# Use the torchrun entry point through the active Python interpreter.
exec python -m torch.distributed.run \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    "$(dirname "$0")/test.py" \
    "$CONFIG" \
    "$CHECKPOINT" \
    --launcher pytorch \
    "${@:4}"
