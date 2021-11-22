#!/bin/sh
NITI_PATH="/media/ssd512g4/maolin/niti"
GPU=0
docker build -t wangmaolin/niti:0.1 .

docker run --shm-size=2g --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$GPU -v $NITI_PATH:/niti -it wangmaolin/niti:0.1
