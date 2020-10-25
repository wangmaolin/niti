NITI_PATH="/media/data2/niti"
docker build -t wangmaolin/niti:0.1 .
docker run --gpus all -v $NITI_PATH:/niti -it wangmaolin/niti:0.1
