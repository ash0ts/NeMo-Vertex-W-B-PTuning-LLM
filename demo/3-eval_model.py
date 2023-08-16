import argparse
import os
from datetime import datetime

import torch
import torch.multiprocessing as mp
import wandb
import wget
from megatron.core import parallel_state
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import \
    MegatronGPTPromptLearningModel
from nemo.collections.nlp.modules.common import VirtualPromptStyle
from nemo.collections.nlp.modules.common.transformer.text_generation import (
    LengthParam, SamplingParam)
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.trainer import Trainer
from utils import download_file


def parse_args():
    parser = argparse.ArgumentParser(description="NeMo Megatron PTuning - Evaluation")

    parser.add_argument("--output_dir", default="output", help="Output directory.")

    # Add argparse parameters for the `inference` block
    # Add argparse parameters for the `inference` block with default values
    parser.add_argument(
        "--greedy",
        type=bool,
        default=False,
        help="Whether or not to use sampling; use greedy decoding otherwise",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--add_BOS",
        type=bool,
        default=True,
        help="Add the bos token at the beginning of the prompt",
    )
    parser.add_argument(
        "--tokens_to_generate",
        type=int,
        default=30,
        help="The minimum length of the sequence to be generated.",
    )
    parser.add_argument(
        "--all_probs",
        type=bool,
        default=False,
        help="Whether return the log prob for all the tokens in vocab",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="The parameter for repetition penalty. 1.0 means no penalty.",
    )
    parser.add_argument(
        "--min_tokens_to_generate",
        type=int,
        default=0,
        help="The minimum length of the sequence to be generated.",
    )
    parser.add_argument(
        "--compute_logprob",
        type=bool,
        default=False,
        help="A flag used to compute logprob of all the input text, a very special case of running inference, default False",
    )
    parser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size for inference"
    )

    # Trainer configs
    parser.add_argument(
        "--accelerator",
        default="gpu" if torch.cuda.is_available() else "cpu",
        help="Accelerator - GPU or CPU.",
    )
    parser.add_argument("--devices", default=1, type=int, help="Number of devices.")
    parser.add_argument(
        "--enable_checkpointing",
        action="store_true",
        help="Whether to enable checkpoints during inference. Default is to not to",
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

    return parser.parse_args()


def main(args):
    run = wandb.init(
        project="NeMo_Megatron_PTuning",
        name=f"eval@{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        config=args,
    )
    args = run.config
    squad_art_path = run.use_artifact("squad:latest", type="datasets").download()
    SQUAD_DIR = os.path.join(squad_art_path, "data", "SQuAD")

    # TODO: Loop all over artifact versions given a filter and use that to apply an alias for model registry
    final_chkpt_path = run.use_artifact(
        "final_model_checkpoints:latest", type="model"
    ).download()
    tuned_model_path = os.path.join(final_chkpt_path, "NeMo_Megatron_PTuning.nemo")
    gpt_model_file = os.path.join(
        final_chkpt_path, "nemo_assets", "megatron_gpt_345m.nemo"
    )

    OUTPUT_DIR = args.output_dir
    NEMO_DIR = os.path.join(OUTPUT_DIR, "nemo_assets")
    CONFIG_DIR = os.path.join(NEMO_DIR, "conf")

    os.makedirs(NEMO_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Download the example config file
    download_file(
        f"https://raw.githubusercontent.com/NVIDIA/NeMo/stable/examples/nlp/language_modeling/conf/megatron_gpt_prompt_learning_inference.yaml",
        CONFIG_DIR,
    )

    # Load the example config file so we can start editing it
    CONFIG_PATH = os.path.join(
        CONFIG_DIR, "megatron_gpt_prompt_learning_inference.yaml"
    )
    cfg = OmegaConf.load(CONFIG_PATH)
    OmegaConf.set_struct(cfg, False)

    # Override configuration values with command line arguments
    cfg.inference.greedy = args.greedy
    cfg.inference.top_k = args.top_k
    cfg.inference.top_p = args.top_p
    cfg.inference.temperature = args.temperature
    cfg.inference.add_BOS = args.add_BOS
    cfg.inference.tokens_to_generate = args.tokens_to_generate
    cfg.inference.all_probs = args.all_probs
    cfg.inference.repetition_penalty = args.repetition_penalty
    cfg.inference.min_tokens_to_generate = args.min_tokens_to_generate
    cfg.inference.compute_logprob = args.compute_logprob
    cfg.inference.batch_size = args.batch_size

    cfg.virtual_prompt_model_file = tuned_model_path
    cfg.model = {
        "language_model_path": gpt_model_file,
        "virtual_prompt_style": VirtualPromptStyle.P_TUNING.value,
    }
    cfg.gpt_model_file = gpt_model_file
    cfg.data_paths = [os.path.join(SQUAD_DIR, "squad_short_test.jsonl")]

    mp.set_start_method("spawn", force=True)

    # let's modify some trainer configs
    # check if we have GPU available and uses it
    accelerator = args.accelerator
    cfg.trainer.accelerator = accelerator
    cfg.trainer.devices = args.devices
    cfg.trainer.enable_checkpointing = args.enable_checkpointing

    # for PyTorch Native AMP set precision=16
    cfg.trainer.precision = args.precision

    # setup cluster environment parameters"
    # use torch elastic cluster environment so `create_process_externally` is True
    # the launcher is set to None. It will not try to spawn new processes.
    # It won't create the misconfiguration error because of the `interactive session`
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is needed for the inference")

    strategy = NLPDDPStrategy(
        find_unused_parameters=False, no_ddp_communication_hook=True
    )
    plugins = [TorchElasticEnvironment()]
    trainer = Trainer(strategy=strategy, plugins=plugins, **cfg.trainer)

    # Set name of the experiment
    cfg.name = args.name
    cfg.exp_manager = {
        "resume_if_exists": args.resume_if_exists,
        "create_wandb_logger": args.create_wandb_logger,
        "wandb_logger_kwargs": {"project": args.project, "log_model": args.log_model},
    }
    OmegaConf.set_struct(cfg, True)

    # Init the experiment manager and view the exp_dir
    exp_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    exp_dir = str(exp_dir)

    if (
        cfg.tensor_model_parallel_size < 0
        or cfg.pipeline_model_parallel_size < 0
        or cfg.get("pipeline_model_parallel_split_rank", -1) < 0
    ):
        model_config = MegatronGPTPromptLearningModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            return_config=True,
        )

        with open_dict(cfg):
            cfg.tensor_model_parallel_size = model_config.get(
                "tensor_model_parallel_size", 1
            )
            cfg.pipeline_model_parallel_size = model_config.get(
                "pipeline_model_parallel_size", 1
            )
            cfg.pipeline_model_parallel_split_rank = model_config.get(
                "pipeline_model_parallel_split_rank", 0
            )

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    # Update frozen GPT model path if it is given in case it has changed
    prompt_learning_cfg = MegatronGPTPromptLearningModel.restore_from(
        cfg.virtual_prompt_model_file,
        trainer=trainer,
        return_config=True,
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
        restore_path=cfg.virtual_prompt_model_file,
        trainer=trainer,
        override_config_path=prompt_learning_cfg,
    )
    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.frozen_model.model.language_model.encoder.activations_checkpoint_method = (
            None
        )
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

    max_seq_length = (
        model.frozen_model.cfg.encoder_seq_length - length_params["max_length"]
    )
    max_seq_length = min(max_seq_length, cfg.get("max_seq_length", 8192))

    _, dataloader = model.build_virtual_prompt_dataset(
        data=cfg.data_paths,
        batch_size=cfg.inference.get("batch_size", 1),
        max_seq_length=max_seq_length,
        min_seq_length=model.cfg.data.get("min_seq_length", 1),
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

    # TODO: Add details from model? Link to model training run?
    prediction_table = wandb.Table(
        columns=["i", "prediction", "context", "question", "answer"]
    )

    def split_text(text):
        split_text = text.split("  ")
        context = [
            i.replace("Context: ", "").strip()
            for i in split_text
            if i.startswith("Context:")
        ]
        question = [
            i.replace("Question: ", "").strip()
            for i in split_text
            if i.startswith("Question:")
        ]
        answer = [
            i.replace("Answer:", "").strip()
            for i in split_text
            if i.startswith("Answer:")
        ]
        return (context, question, answer)

    print("***************************")
    for i in range(len(response)):
        for sent in response[i]["sentences"]:
            sent = sent.strip()
            sent = sent.replace("\n", " ")
            context, question, answer = split_text(sent)
        prediction_table.add_data(i, sent, context, question, answer)
    print("***************************")

    wandb.log({"prediction_table": prediction_table})
    wandb.run.log_code()
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
