import wandb
import os
import wget
from datetime import datetime
from omegaconf import OmegaConf
from nemo.collections.nlp.modules.common import VirtualPromptStyle
import torch
import torch.multiprocessing as mp
from megatron.core import parallel_state
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
    MegatronGPTPromptLearningModel,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from nemo.utils.exp_manager import exp_manager

def main():
    run = wandb.init(
        project="NeMo_Megatron_PTuning",
        name=f"eval@{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    squad_art_path = run.use_artifact("squad:latest", type="datasets").download()
    SQUAD_DIR = os.path.join(squad_art_path, "data", "SQuAD")

    #TODO: Loop all over artifact versions given a filter and use that to apply an alias for model registry
    final_chkpt_path = run.use_artifact("final_model_checkpoints:latest", type="model").download()
    tuned_model_path = os.path.join(final_chkpt_path, "NeMo_Megatron_PTuning.nemo")
    gpt_model_file = os.path.join(final_chkpt_path, "nemo_assets", "megatron_gpt_345m.nemo")

    OUTPUT_DIR = "output"
    NEMO_DIR = os.path.join(OUTPUT_DIR, "nemo_assets")

    os.makedirs(NEMO_DIR, exist_ok=True)

    CONFIG_DIR = os.path.join(NEMO_DIR, "conf")
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Download the example config file
    wget.download(f'https://raw.githubusercontent.com/NVIDIA/NeMo/stable/examples/nlp/language_modeling/conf/megatron_gpt_prompt_learning_inference.yaml', CONFIG_DIR)

    # Load the example config file so we can start editing it
    CONFIG_PATH = os.path.join(CONFIG_DIR, "megatron_gpt_prompt_learning_inference.yaml")
    cfg = OmegaConf.load(CONFIG_PATH)
    OmegaConf.set_struct(cfg, False)

    cfg.virtual_prompt_model_file = tuned_model_path
    cfg.model = {
        "language_model_path": gpt_model_file,
        "virtual_prompt_style": VirtualPromptStyle.P_TUNING.value,
    }
    cfg.gpt_model_file =  gpt_model_file
    cfg.data_paths = [os.path.join(SQUAD_DIR,"squad_short_test.jsonl")]
    #TODO: Remove and write to wandb.Table
    cfg.pred_file_path = "predictions.txt"

    mp.set_start_method("spawn", force=True)

    # let's modify some trainer configs
    # check if we have GPU available and uses it
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    cfg.trainer.accelerator = accelerator
    cfg.trainer.devices = 1
    cfg.trainer.enable_checkpointing = False

    # for PyTorch Native AMP set precision=16
    cfg.trainer.precision = 16 if torch.cuda.is_available() else 32

    # setup cluster environment parameters"
    # use torch elastic cluster environment so `create_process_externally` is True
    # the launcher is set to None. It will not try to spawn new processes.
    # It won't create the misconfiguration error because of the `interactive session`
    os.environ["LOCAL_RANK"] = '0'
    os.environ["RANK"] = '0'
    os.environ["WORLD_SIZE"] = '1'

    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is needed for the inference")

    strategy = NLPDDPStrategy(find_unused_parameters=False, no_ddp_communication_hook=True)
    plugins = [TorchElasticEnvironment()]
    trainer = Trainer(strategy=strategy, plugins=plugins, **cfg.trainer)

    # Set name of the experiment
    cfg.name = 'NeMo_Megatron_PTuning'
    cfg.exp_manager = {
        "resume_if_exists" : False,
        "create_wandb_logger" : True,
        "wandb_logger_kwargs" :
            {
                "project" : "NeMo_Megatron_PTuning",
                "log_model" : "all"
            }
    }
    OmegaConf.set_struct(cfg, True)

    # Init the experiment manager and view the exp_dir
    exp_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    exp_dir = str(exp_dir)

    if (
        cfg.tensor_model_parallel_size < 0
        or cfg.pipeline_model_parallel_size < 0
        or cfg.get('pipeline_model_parallel_split_rank', -1) < 0
    ):
        model_config = MegatronGPTPromptLearningModel.restore_from(
            restore_path=cfg.gpt_model_file, trainer=trainer, return_config=True,
        )

        with open_dict(cfg):
            cfg.tensor_model_parallel_size = model_config.get('tensor_model_parallel_size', 1)
            cfg.pipeline_model_parallel_size = model_config.get('pipeline_model_parallel_size', 1)
            cfg.pipeline_model_parallel_split_rank = model_config.get('pipeline_model_parallel_split_rank', 0)

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    # Update frozen GPT model path if it is given in case it has changed
    prompt_learning_cfg = MegatronGPTPromptLearningModel.restore_from(
        cfg.virtual_prompt_model_file, trainer=trainer, return_config=True,
    )
    if cfg.get("gpt_model_file"):
        with open_dict(prompt_learning_cfg):
            prompt_learning_cfg.language_model_path = cfg.gpt_model_file
            prompt_learning_cfg.sequence_parallel = False
            prompt_learning_cfg.activations_checkpoint_method = None
            prompt_learning_cfg.activations_checkpoint_granularity = None
            prompt_learning_cfg.activations_checkpoint_num_layers = None
            prompt_learning_cfg.virtual_prompt_style = cfg.model.virtual_prompt_style
    
    # Load prompt tuned model, virtual_prompt_model_file must be provided in config
    # Now load prompt learning model with frozen gpt model base
    model = MegatronGPTPromptLearningModel.restore_from(
        restore_path=cfg.virtual_prompt_model_file, trainer=trainer, override_config_path=prompt_learning_cfg,
    )
    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.frozen_model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    # Check whether the DDP is initialized
    if parallel_state.is_unitialized():

        def placeholder():
            return

        if model.trainer.strategy.launcher is not None:
            model.trainer.strategy.launcher.launch(placeholder, trainer=model.trainer)
        model.trainer.strategy.setup_environment()

    length_params: LengthParam = {
        "max_length": cfg.inference.tokens_to_generate,
        "min_length": cfg.inference.min_tokens_to_generate,
    }

    sampling_params: SamplingParam = {
        "use_greedy": cfg.inference.greedy,
        "temperature": cfg.inference.temperature,
        "top_k": cfg.inference.top_k,
        "top_p": cfg.inference.top_p,
        "repetition_penalty": cfg.inference.repetition_penalty,
        "add_BOS": cfg.inference.add_BOS,
        "all_probs": cfg.inference.all_probs,
        "compute_logprob": cfg.inference.compute_logprob,
    }

    max_seq_length = model.frozen_model.cfg.encoder_seq_length - length_params["max_length"]
    max_seq_length = min(max_seq_length, cfg.get("max_seq_length", 8192))

    _, dataloader = model.build_virtual_prompt_dataset(
        data=cfg.data_paths,
        batch_size=cfg.inference.get('batch_size', 1),
        max_seq_length=max_seq_length,
        min_seq_length=model.cfg.data.get('min_seq_length', 1),
        add_bos=sampling_params["add_BOS"],
        add_eos=False,
        for_train=False,
        tokens_to_generate=length_params["max_length"],
        drop_last=False,
        shuffle=False,
        num_workers=cfg.get("num_workers", 1),
    )

    config = OmegaConf.to_container(cfg.inference)
    model.set_inference_config(config)

    response = trainer.predict(model, dataloader)

    #TODO: Add details from model? Link to model training run?
    prediction_table = wandb.Table(columns=["i", "prediction", "context", "question", "answer"])

    def split_text(text):
        split_text = text.split('  ')
        context = [i.replace('Context: ', '').strip() for i in split_text if i.startswith('Context:')]
        question = [i.replace('Question: ', '').strip() for i in split_text if i.startswith('Question:')]
        answer = [i.replace('Answer:', '').strip() for i in split_text if i.startswith('Answer:')]
        return (context, question, answer)
    
    print("***************************")
    for i in range(len(response)):
        for sent in response[i]["sentences"]:
            sent = sent.strip()
            sent = sent.replace("\n", " ")
            context, question, answer = split_text(sent)
        prediction_table.add_data(i, sent, context, question, answer)
    print("***************************")

    wandb.log(
        {
            "prediction_table": prediction_table
        }
    )
    wandb.finish()

if __name__ == "__main__":
    main()