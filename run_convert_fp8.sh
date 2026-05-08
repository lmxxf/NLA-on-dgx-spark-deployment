#!/bin/bash
# 双机 FP8 量化启动脚本
# 在 host (spark-3a10) 上执行

set -e

IMAGE="nla-llama70b:base"
HEAD_IP="169.254.248.35"
WORKER_SSH="lmxxf@169.254.30.81"
MASTER_PORT=29500
ETH_IF="enp1s0f1np1"
IB_HCA="rocep1s0f1"

WORKSPACE="/home/lmxxf/work/natural_language_autoencoders"
CONTAINER_HEAD="fp8-head"
CONTAINER_WORKER="fp8-worker"

NCCL_ENVS="\
  -e NCCL_SOCKET_IFNAME=$ETH_IF \
  -e NCCL_IB_HCA=$IB_HCA \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e GLOO_SOCKET_IFNAME=$ETH_IF \
  -e NCCL_DEBUG=WARN"

echo "=== Syncing script to slave ==="
rsync -a $WORKSPACE/convert_fp8_dual.py $WORKER_SSH:$WORKSPACE/

echo "=== Stopping old containers ==="
docker rm -f $CONTAINER_HEAD 2>/dev/null || true
ssh $WORKER_SSH "docker rm -f $CONTAINER_WORKER 2>/dev/null" || true

echo "=== Starting worker (rank=1) on slave ==="
ssh $WORKER_SSH "docker run -d --gpus all \
  --name $CONTAINER_WORKER \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  -v /home/lmxxf/work:/work \
  -e MASTER_ADDR=$HEAD_IP \
  -e MASTER_PORT=$MASTER_PORT \
  -e WORLD_SIZE=2 \
  -e RANK=1 \
  -e LOCAL_RANK=0 \
  $NCCL_ENVS \
  $IMAGE \
  python3 /work/natural_language_autoencoders/convert_fp8_dual.py"

echo "=== Starting head (rank=0) on host ==="
docker run --rm --gpus all \
  --name $CONTAINER_HEAD \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  -v /home/lmxxf/work:/work \
  -e MASTER_ADDR=$HEAD_IP \
  -e MASTER_PORT=$MASTER_PORT \
  -e WORLD_SIZE=2 \
  -e RANK=0 \
  -e LOCAL_RANK=0 \
  $NCCL_ENVS \
  $IMAGE \
  python3 /work/natural_language_autoencoders/convert_fp8_dual.py

echo "=== Worker logs ==="
ssh $WORKER_SSH "docker logs $CONTAINER_WORKER 2>&1 | tail -10"
ssh $WORKER_SSH "docker rm -f $CONTAINER_WORKER 2>/dev/null" || true

echo "=== Syncing slave shards back to host ==="
rsync -avP $WORKER_SSH:/home/lmxxf/work/models/Llama-3.3-70B-NLA-L53-av-fp8/*.safetensors \
  /home/lmxxf/work/models/Llama-3.3-70B-NLA-L53-av-fp8/

echo "=== Building index.json and config.json ==="
docker run --rm \
  -v /home/lmxxf/work:/work \
  $IMAGE \
  python3 /work/natural_language_autoencoders/build_fp8_index.py

echo "=== Done ==="
