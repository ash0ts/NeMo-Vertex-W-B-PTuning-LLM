import wandb
from datetime import datetime
import wget
import os
from omegaconf import OmegaConf
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
import subprocess
from nemo.collections.nlp.modules.common import VirtualPromptStyle
import apex
import torch
import pytorch_lightning as pl
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from nemo.utils.exp_manager import exp_manager
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import MegatronGPTPromptLearningModel
from pathlib import Path

def main():
    run = wandb.init(
        project="NeMo_Megatron_PTuning",
        name=f"train@{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    squad_art_path = run.use_artifact("squad:latest", type="datasets").download()
    SQUAD_DIR = os.path.join(squad_art_path, "data", "SQuAD")
    OUTPUT_DIR = "output"
    NEMO_DIR = os.path.join(OUTPUT_DIR, "nemo_assets")
    CONFIG_DIR = os.path.join(OUTPUT_DIR, "conf")

    os.makedirs(NEMO_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Download the example config file
    wget.download(f'https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/nlp/language_modeling/conf/megatron_gpt_prompt_learning_config.yaml', CONFIG_DIR)

    # Load the example config file so we can start editing it
    CONFIG_PATH = os.path.join(CONFIG_DIR, "megatron_gpt_prompt_learning_config.yaml")
    config = OmegaConf.load(CONFIG_PATH)

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
    subprocess.run(["wget", "-nc", "--content-disposition", url, "-O", gpt_file_path], check=True)
    
    config.model.language_model_path = gpt_file_path

    config.exp_manager.checkpoint_callback_params.save_nemo_on_train_end= True
    config.exp_manager.checkpoint_callback_params.always_save_nemo= True
    config.exp_manager.checkpoint_callback_params.save_best_model= True

    config.model.virtual_prompt_style = VirtualPromptStyle.P_TUNING
    config.model.p_tuning.dropout = 0.0
    config.model.p_tuning.num_layers = 2
    config.model.global_batch_size = 2
    config.model.micro_batch_size = 1

    # let's modify some trainer configs
    # check if we have GPU available and uses it
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    config.trainer.accelerator = accelerator
    config.trainer.devices = 1
    config.trainer.max_epochs = 4
    config.trainer.val_check_interval = 1.0

    # for PyTorch Native AMP set precision=16
    config.trainer.precision = 16 if torch.cuda.is_available() else 32

    # setup cluster environment parameters"
    # use torch elastic cluster environment so `create_process_externally` is True
    # the launcher is set to None. It will not try to spawn new processes.
    # It won't create the misconfiguration error because of the `interactive session`
    os.environ["LOCAL_RANK"] = '0'
    os.environ["RANK"] = '0'
    os.environ["WORLD_SIZE"] = '1'

    strategy = NLPDDPStrategy(find_unused_parameters=False, no_ddp_communication_hook=True)
    plugins = [TorchElasticEnvironment()]
    trainer = pl.Trainer(plugins= plugins, strategy=strategy, **config.trainer)

    # Set name of the experiment
    config.name = 'NeMo_Megatron_PTuning'
    config.exp_manager.resume_if_exists = False
    config.exp_manager.create_wandb_logger = True
    config.exp_manager.wandb_logger_kwargs.project = "NeMo_Megatron_PTuning"
    config.exp_manager.wandb_logger_kwargs.log_model = "all"

    # Init the experiment manager and view the exp_dir
    exp_dir = exp_manager(trainer, config.get("exp_manager", None))
    exp_dir = str(exp_dir)

    # Set some of the learning parameters
    config.model.optim.lr = 1e-4
    config.model.precision = config.trainer.precision

    model = MegatronGPTPromptLearningModel(cfg=config.model, trainer=trainer)
    # Training set to 2 epochs by default in a cell above
    # Each epoch will take around 1min 15sec, but training time can vary
    trainer.fit(model)

    test_examples = [
        {"taskname": "squad", "context": "The build was released for download later in the day in standard 32-bit and 64-bit versions, plus a special 64-bit version which included SDKs and developer tools (Visual Studio Express and Expression Blend) for developing Metro-style apps. The Windows Store was announced during the presentation, but was not available in this build. According to Microsoft, there were about 535,000 downloads of the developer preview within the first 12 hours of its release. Originally set to expire on March 11, 2012, in February 2012 the Developer Preview's expiry date was changed to January 15, 2013.", "question": "When was the Developer preview initially intended to expire?"},
        {"taskname": "squad", "context": "The structures of most federal governments incorporate mechanisms to protect the rights of component states. One method, known as 'intrastate federalism', is to directly represent the governments of component states in federal political institutions. Where a federation has a bicameral legislature the upper house is often used to represent the component states while the lower house represents the people of the nation as a whole. A federal upper house may be based on a special scheme of apportionment, as is the case in the senates of the United States and Australia, where each state is represented by an equal number of senators irrespective of the size of its population.", "question": "What is a bicameral legislature?"},
        {"taskname": "squad", "context": "Imported mystery religions, which offered initiates salvation in the afterlife, were a matter of personal choice for an individual, practiced in addition to carrying on one's family rites and participating in public religion. The mysteries, however, involved exclusive oaths and secrecy, conditions that conservative Romans viewed with suspicion as characteristic of \"magic\", conspiratorial (coniuratio), or subversive activity. Sporadic and sometimes brutal attempts were made to suppress religionists who seemed to threaten traditional morality and unity, as with the senate's efforts to restrict the Bacchanals in 186 BC.", "question": "What was the practice of religion to the Romans?"}
    ]

    response = model.generate(inputs=test_examples, length_params=None)

    print('The prediction results of some sample queries with the trained model:')
    for result in response['sentences']:
        print(result)
        print("-" * 30)
    
    trained_model_chkpt = Path(exp_dir, "checkpoints", config.model.nemo_path)
    final_chkpt = wandb.Artifact(name="final_model_checkpoints", type="model")
    final_chkpt.add_file(trained_model_chkpt)
    final_chkpt.add_dir(NEMO_DIR, name="nemo_assets")
    wandb.log_artifact(final_chkpt)
    wandb.finish()

if __name__ == "__main__":
    main()