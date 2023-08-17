#!/bin/bash

# Desired CUDA version
CUDA_VERSION="12.1"
CUDA_PATH="/usr/local/cuda-$CUDA_VERSION"

# Update PATH
export PATH=$CUDA_PATH/bin:$PATH

# Update LD_LIBRARY_PATH for dynamic linking
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Optionally set CUDA_HOME
export CUDA_HOME=$CUDA_PATH

echo "Environment set for CUDA $CUDA_VERSION!"
