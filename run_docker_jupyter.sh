#!/bin/bash

docker build . -t nvidia-vertex-wb-demo
docker run --runtime=nvidia -it --rm -v --shm-size=16g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvidia-vertex-wb-demo jupyter lab --NotebookApp.token=''