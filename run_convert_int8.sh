#!/bin/bash
# 双机 INT8 量化启动脚本（在 host 上执行）
set -e

IMAGE="nla-llama70b:base"
WORKER_SSH="lmxxf@169.254.30.81"
WORKSPACE="/home/lmxxf/work/NLA-on-dgx-spark-deployment"

echo "=== Syncing scripts to slave ==="
rsync -aP $WORKSPACE/convert_int8_dual.py $WORKER_SSH:$WORKSPACE/

echo "=== Stopping old containers ==="
docker rm -f int8-head 2>/dev/null || true
ssh $WORKER_SSH "docker rm -f int8-worker 2>/dev/null" || true

echo "=== Starting worker (rank=1) on slave ==="
ssh $WORKER_SSH "docker run -d --gpus all \
  --name int8-worker \
  --network host --ipc host \
  -v /home/lmxxf/work:/work \
  -e RANK=1 -e WORLD_SIZE=2 \
  $IMAGE \
  python3 /work/NLA-on-dgx-spark-deployment/convert_int8_dual.py"

echo "=== Starting head (rank=0) on host ==="
docker run --rm --gpus all \
  --name int8-head \
  --network host --ipc host \
  -v /home/lmxxf/work:/work \
  -e RANK=0 -e WORLD_SIZE=2 \
  $IMAGE \
  python3 /work/NLA-on-dgx-spark-deployment/convert_int8_dual.py

echo "=== Worker logs ==="
ssh $WORKER_SSH "docker logs int8-worker 2>&1 | tail -10"
ssh $WORKER_SSH "docker rm -f int8-worker 2>/dev/null" || true

echo "=== Syncing slave shards to host ==="
rsync -avP $WORKER_SSH:/home/lmxxf/work/models/Llama-3.3-70B-NLA-L53-av-int8/*.safetensors \
  /home/lmxxf/work/models/Llama-3.3-70B-NLA-L53-av-int8/

echo "=== Building index and config ==="
docker run --rm \
  -v /home/lmxxf/work:/work \
  $IMAGE \
  python3 /work/NLA-on-dgx-spark-deployment/build_int8_index.py

echo "=== Done ==="
du -sh /home/lmxxf/work/models/Llama-3.3-70B-NLA-L53-av-int8/
