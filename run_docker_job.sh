#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Please provide the Python file to run as an argument."
    exit 1
fi

# Build the Docker image
# docker build . -t nvidia-vertex-wb-demo --no-cache
docker build . -t nvidia-vertex-wb-demo

# # Tag and push the Docker image to Docker Hub
docker tag nvidia-vertex-wb-demo:latest ash0ts/nvidia-vertex-wb-demo:latest
docker push ash0ts/nvidia-vertex-wb-demo:latest

# Run the specified Python file in the container
docker run \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_DOCKER="ash0ts/nvidia-vertex-wb-demo" \
    --runtime=nvidia -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 \
    ash0ts/nvidia-vertex-wb-demo \
    python $1
