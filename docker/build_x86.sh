#!/bin/bash
. docker/env.bash
docker build \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    --build-arg BASE_IMAGE=$BASE_IMAGE \
    --build-arg UPDATE_NV_KEYS=true \
    --build-arg WORKDIR=$WORKDIR \
    -t $CONTAINER_NAME \
    -f Dockerfile .
