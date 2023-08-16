import argparse
import os
import subprocess
from datetime import datetime

import wandb
from utils import download_file, subset_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="NeMo Megatron PTuning - Data Prep")

    # Directories
    parser.add_argument("--output_dir", default="output", help="Output directory.")

    return parser.parse_args()


def main(args):
    run = wandb.init(
        project="NeMo_Megatron_PTuning",
        name=f"data_prep_squad@{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        config=args,
    )
    args = run.config

    # You can replace DATA_DIR and NEMO_DIR with your own locations
    OUTPUT_DIR = args.output_dir
    DATA_DIR = os.path.join(OUTPUT_DIR, "data")
    NEMO_DIR = os.path.join(OUTPUT_DIR, "nemo_assets")
    SQUAD_DIR = os.path.join(DATA_DIR, "SQuAD")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(NEMO_DIR, exist_ok=True)
    os.makedirs(SQUAD_DIR, exist_ok=True)

    # download the preprocessing scripts from github for the purpose of this tutorial
    download_file(
        f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/dataset_processing/nlp/squad/prompt_learning_squad_preprocessing.py",
        str(NEMO_DIR),
    )

    # Download the SQuAD dataset
    download_file(
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
        str(SQUAD_DIR),
    )
    download_file(
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
        str(SQUAD_DIR),
    )

    # Preprocess squad data
    subprocess.run(
        [
            "python",
            f"{NEMO_DIR}/prompt_learning_squad_preprocessing.py",
            "--data-dir",
            SQUAD_DIR,
        ]
    )

    # Using the function for your files
    subset_jsonl(
        f"{SQUAD_DIR}/squad_train.jsonl", f"{SQUAD_DIR}/squad_short_train.jsonl", 2000
    )
    subset_jsonl(
        f"{SQUAD_DIR}/squad_val.jsonl", f"{SQUAD_DIR}/squad_short_val.jsonl", 200
    )
    subset_jsonl(
        f"{SQUAD_DIR}/squad_test.jsonl", f"{SQUAD_DIR}/squad_short_test.jsonl", 2000
    )

    data_artifact = wandb.Artifact(name="squad", type="datasets")
    data_artifact.add_dir(OUTPUT_DIR)
    run.log_artifact(data_artifact)
    run.log_code()
    run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
