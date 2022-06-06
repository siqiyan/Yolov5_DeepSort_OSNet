#!/bin/bash
. docker/env.bash
xhost +si:localuser:root
if [ ! -d "data" ]; then
    mkdir -p data
fi
docker run --rm --gpus all --privileged --net host \
    --name $CONTAINER_NAME \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -it $CONTAINER_NAME \
    /bin/bash
