import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path

import apex
import pytorch_lightning as pl
import torch
import wandb
import wget
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import \
    MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import \
    MegatronGPTPromptLearningModel
from nemo.collections.nlp.modules.common import VirtualPromptStyle
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from utils import download_file


def parse_args():
    parser = argparse.ArgumentParser(description="NeMo Megatron PTuning - Training")

    # Directories
    parser.add_argument("--output_dir", default="output", help="Output directory.")

    # Model configs
    parser.add_argument(
        "--global_batch_size", default=8, type=int, help="Global batch size."
    )
    parser.add_argument(
        "--micro_batch_size", default=4, type=int, help="Micro batch size."
    )
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument(
        "--num_layers", default=2, type=int, help="Number of layers for p-tuning."
    )
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")

    # Trainer configs
    parser.add_argument(
        "--accelerator",
        default="gpu" if torch.cuda.is_available() else "cpu",
        help="Accelerator - GPU or CPU.",
    )
    parser.add_argument("--devices", default=1, type=int, help="Number of devices.")
    parser.add_argument(
        "--max_epochs", default=10, type=int, help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--val_check_interval",
        default=1.0,
        type=float,
        help="Validation check interval.",
    )
    parser.add_argument(
        "--precision",
        default=16 if torch.cuda.is_available() else 32,
        type=int,
        help="Training precision.",
    )

    # Experiment manager configs
    parser.add_argument(
        "--name", default="NeMo_Megatron_PTuning", help="Name of the experiment."
    )
    parser.add_argument(
        "--resume_if_exists",
        action="store_true",
        help="Whether to resume if exists. Default is to not resume",
    )
    parser.add_argument(
        "--no_create_wandb_logger",
        action="store_false",
        dest="create_wandb_logger",
        help="Specify this flag to not create wandb logger. Default is to create wandb logger.",
    )
    parser.add_argument(
        "--project", default="NeMo_Megatron_PTuning", help="WandB project name."
    )
    parser.add_argument("--log_model", default="all", help="Log model in WandB.")

    # Checkpoint management
    parser.add_argument(
        "--save_nemo_on_train_end",
        type=bool,
        default=True,
        help="Whether to save nemo model on train end.",
    )
    parser.add_argument(
        "--always_save_nemo",
        type=bool,
        default=True,
        help="Whether to always save the nemo model.",
    )
    parser.add_argument(
        "--save_best_model",
        type=bool,
        default=True,
        help="Whether to save the best model during training.",
    )

    return parser.parse_args()


def main(args):
    run = wandb.init(
        project="NeMo_Megatron_PTuning",
        name=f"train@{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        config=args,
    )
    args = run.config
    squad_art_path = run.use_artifact("squad:latest", type="datasets").download()
    SQUAD_DIR = os.path.join(squad_art_path, "data", "SQuAD")
    OUTPUT_DIR = args.output_dir
    NEMO_DIR = os.path.join(OUTPUT_DIR, "nemo_assets")
    CONFIG_DIR = os.path.join(OUTPUT_DIR, "conf")

    os.makedirs(NEMO_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Download the example config file
    download_file(
        f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/nlp/language_modeling/conf/megatron_gpt_prompt_learning_config.yaml",
        CONFIG_DIR,
    )

    # Load the example config file so we can start editing it
    CONFIG_PATH = os.path.join(CONFIG_DIR, "megatron_gpt_prompt_learning_config.yaml")
    config = OmegaConf.load(CONFIG_PATH)

    config.model.p_tuning.dropout = args.dropout
    config.model.p_tuning.num_layers = args.num_layers
    config.model.global_batch_size = args.global_batch_size
    config.model.micro_batch_size = args.micro_batch_size

    accelerator = args.accelerator
    config.trainer.accelerator = accelerator
    config.trainer.devices = args.devices
    config.trainer.max_epochs = args.max_epochs
    config.trainer.val_check_interval = args.val_check_interval
    config.trainer.precision = args.precision

    config.name = args.name
    config.exp_manager.resume_if_exists = args.resume_if_exists
    config.exp_manager.create_wandb_logger = args.create_wandb_logger
    config.exp_manager.wandb_logger_kwargs.project = args.project
    config.exp_manager.wandb_logger_kwargs.log_model = args.log_model

    config.exp_manager.checkpoint_callback_params.save_nemo_on_train_end = (
        args.save_nemo_on_train_end
    )
    config.exp_manager.checkpoint_callback_params.always_save_nemo = (
        args.always_save_nemo
    )
    config.exp_manager.checkpoint_callback_params.save_best_model = args.save_best_model

    # Set some of the learning parameters
    config.model.optim.lr = args.lr
    config.model.precision = config.trainer.precision

    # TODO: Parameterize the Task specific information
    config.model.virtual_prompt_style = VirtualPromptStyle.P_TUNING
    config.model.data.train_ds = [f"{SQUAD_DIR}/squad_short_train.jsonl"]
    config.model.data.validation_ds = [f"{SQUAD_DIR}/squad_short_val.jsonl"]
    config.model.task_templates = [
        {
            "taskname": "squad",
            "prompt_template": "<|VIRTUAL_PROMPT_0|> Context: {context}\n\nQuestion: {question}\n\nAnswer:{answer}",
            "total_virtual_tokens": 15,
            "virtual_token_splits": [15],
            "truncate_field": "context",
            "answer_only_loss": True,
            "answer_field": "answer",
        },
    ]
    config.model.existing_tasks = []
    config.model.new_tasks = ["squad"]

    # Download the model from NGC
    gpt_file_name = "megatron_gpt_345m.nemo"
    gpt_file_path = os.path.join(NEMO_DIR, gpt_file_name)
    url = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/megatron_gpt_345m/versions/1/files/megatron_gpt_345m.nemo"
    if not os.path.exists(gpt_file_path):
        subprocess.run(
            ["wget", "-nc", "--content-disposition", url, "-O", gpt_file_path],
            check=True,
        )

    config.model.language_model_path = gpt_file_path

    # setup cluster environment parameters"
    # use torch elastic cluster environment so `create_process_externally` is True
    # the launcher is set to None. It will not try to spawn new processes.
    # It won't create the misconfiguration error because of the `interactive session`
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    strategy = NLPDDPStrategy(
        find_unused_parameters=False, no_ddp_communication_hook=True
    )
    plugins = [TorchElasticEnvironment()]
    # BUG: issue with use_distributed_sampler
    del config.trainer.use_distributed_sampler
    trainer = pl.Trainer(plugins=plugins, strategy=strategy, **config.trainer)

    # Init the experiment manager and view the exp_dir
    exp_dir = exp_manager(trainer, config.get("exp_manager", None))
    exp_dir = str(exp_dir)

    model = MegatronGPTPromptLearningModel(cfg=config.model, trainer=trainer)
    # Training set to 2 epochs by default in a cell above
    # Each epoch will take around 1min 15sec, but training time can vary
    trainer.fit(model)

    test_examples = [
        {
            "taskname": "squad",
            "context": "The build was released for download later in the day in standard 32-bit and 64-bit versions, plus a special 64-bit version which included SDKs and developer tools (Visual Studio Express and Expression Blend) for developing Metro-style apps. The Windows Store was announced during the presentation, but was not available in this build. According to Microsoft, there were about 535,000 downloads of the developer preview within the first 12 hours of its release. Originally set to expire on March 11, 2012, in February 2012 the Developer Preview's expiry date was changed to January 15, 2013.",
            "question": "When was the Developer preview initially intended to expire?",
        },
        {
            "taskname": "squad",
            "context": "The structures of most federal governments incorporate mechanisms to protect the rights of component states. One method, known as 'intrastate federalism', is to directly represent the governments of component states in federal political institutions. Where a federation has a bicameral legislature the upper house is often used to represent the component states while the lower house represents the people of the nation as a whole. A federal upper house may be based on a special scheme of apportionment, as is the case in the senates of the United States and Australia, where each state is represented by an equal number of senators irrespective of the size of its population.",
            "question": "What is a bicameral legislature?",
        },
        {
            "taskname": "squad",
            "context": "Imported mystery religions, which offered initiates salvation in the afterlife, were a matter of personal choice for an individual, practiced in addition to carrying on one's family rites and participating in public religion. The mysteries, however, involved exclusive oaths and secrecy, conditions that conservative Romans viewed with suspicion as characteristic of \"magic\", conspiratorial (coniuratio), or subversive activity. Sporadic and sometimes brutal attempts were made to suppress religionists who seemed to threaten traditional morality and unity, as with the senate's efforts to restrict the Bacchanals in 186 BC.",
            "question": "What was the practice of religion to the Romans?",
        },
    ]

    response = model.generate(inputs=test_examples, length_params=None)

    print("The prediction results of some sample queries with the trained model:")
    for result in response["sentences"]:
        print(result)
        print("-" * 30)

    trained_model_chkpt = Path(exp_dir, "checkpoints", config.model.nemo_path)
    final_chkpt = wandb.Artifact(name="final_model_checkpoints", type="model")
    final_chkpt.add_file(trained_model_chkpt)
    final_chkpt.add_dir(NEMO_DIR, name="nemo_assets")
    wandb.log_artifact(final_chkpt)
    wandb.run.log_code()
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
