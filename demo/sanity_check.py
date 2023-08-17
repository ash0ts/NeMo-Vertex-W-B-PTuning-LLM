import set_cuda_env
set_cuda_env.set_environment()
import torch
import subprocess
subprocess.run("./set_cuda.sh")
subprocess.run("nvidia-smi")
print(torch.cuda.is_available())
print(torch.cuda.current_device())