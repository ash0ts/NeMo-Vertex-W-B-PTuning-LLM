# Check if at least one argument is provided
if [ "$#" -eq 0 ]; then
    echo "Please provide the args to the Python module"
    exit 1
fi

# Use "$@" to capture and pass all arguments
wandb launch -j a-sh0ts/NeMo_Megatron_PTuning-demo/job-ash0ts_nvidia-vertex-wb-demo:latest -q anish-nvidia-vm -e a-sh0ts -E "/opt/nvidia/nvidia_entrypoint.sh python 3-eval_model.py $@" -R resource_args.json
