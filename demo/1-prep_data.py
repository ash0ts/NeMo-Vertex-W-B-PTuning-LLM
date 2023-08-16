import wandb
import os
import wget
from datetime import datetime
import subprocess

def main():
    run = wandb.init(
        project="NeMo_Megatron_PTuning",
        name=f"data_prep_squad@{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    # You can replace DATA_DIR and NEMO_DIR with your own locations
    OUTPUT_DIR = "output"
    DATA_DIR = os.path.join(OUTPUT_DIR, "data")
    NEMO_DIR = os.path.join(OUTPUT_DIR, "nemo_assets")
    SQUAD_DIR = os.path.join(DATA_DIR, "SQuAD")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(NEMO_DIR, exist_ok=True)
    os.makedirs(SQUAD_DIR, exist_ok=True)

    # download the preprocessing scripts from github for the purpose of this tutorial
    wget.download(f'https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/dataset_processing/nlp/squad/prompt_learning_squad_preprocessing.py', str(NEMO_DIR))
    
    # Download the SQuAD dataset
    wget.download("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json", str(SQUAD_DIR))
    wget.download("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json", str(SQUAD_DIR))
    
    # Preprocess squad data
    subprocess.run(["python", f"{NEMO_DIR}/prompt_learning_squad_preprocessing.py", "--data-dir", SQUAD_DIR], check=True)

    subprocess.run(["head", "-2000", f"{SQUAD_DIR}/squad_train.jsonl", ">", f"{SQUAD_DIR}/squad_short_train.jsonl"], shell=True, check=True)
    subprocess.run(["head", "-200", f"{SQUAD_DIR}/squad_val.jsonl", ">", f"{SQUAD_DIR}/squad_short_val.jsonl"], shell=True, check=True)
    subprocess.run(["head", "-2000", f"{SQUAD_DIR}/squad_test.jsonl", ">", f"{SQUAD_DIR}/squad_short_test.jsonl"], shell=True, check=True)

    data_artifact = wandb.Artifact(name="squad", type="datasets")
    data_artifact.add_dir(OUTPUT_DIR)
    run.log_artifact(data_artifact)
    run.finish()

if __name__ == "__main__":
    main()