docker build . -t nvidia-vertex-wb-demo --no-cache
docker run -d --runtime=nvidia --ipc=host --shm-size=16g --gpus=all -v /home/temp/jupyter-data:/data --net=host -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvidia-vertex-wb-demo