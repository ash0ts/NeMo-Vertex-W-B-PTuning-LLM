#!/bin/bash

# Set the path where CUDA directories are usually found
CUDA_PATH="/usr/local"

# Find and print all CUDA versions along with their paths
find $CUDA_PATH -maxdepth 1 -type d -name "cuda*" | while read dir; do
    version=$(basename $dir | cut -d "-" -f 2)
    if [[ ! -z $version ]]; then
        echo "Found CUDA version: $version at $dir"
    fi
done
