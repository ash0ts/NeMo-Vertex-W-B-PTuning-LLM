# Using NVIDIA's NeMo image as a base
FROM nvcr.io/ea-bignlp/nemofw-training:23.07-py3

# Copy the "demo" directory into the container
COPY demo demo

# Set the working directory inside the container
WORKDIR demo

# Make set_cuda.sh executable
RUN chmod +x ./set_cuda.sh
RUN ./set_cuda.sh

# Install additional Python packages
RUN pip install ujson
RUN pip install evaluate
RUN pip install --upgrade wandb
# RUN pip install light-the-torch
# RUN pip uninstall -y torch torchvision torchaudio torchdata torchmetrics
# RUN ltt install torch torchvision torchaudio torchdata torchmetrics