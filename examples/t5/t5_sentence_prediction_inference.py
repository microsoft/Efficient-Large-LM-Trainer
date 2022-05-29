import logging
import os
import sys
from argparse import Namespace

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf, open_dict

from fairseq import checkpoint_utils, distributed_utils, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import add_defaults, hydra_init
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.dataclass.utils import omegaconf_no_object_check
from fairseq.logging import progress_bar
from fairseq.tasks.t5_sentence_prediction import T5SentencePredictionTask
from fairseq.utils import reset_logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")


def main(cfg: DictConfig, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)
    add_defaults(cfg)

    if (
        distributed_utils.is_master(cfg.distributed_training)
        and "job_logging_cfg" in cfg
    ):
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    logger.info(cfg)

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0
    assert data_parallel_world_size == 1

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    task: T5SentencePredictionTask
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(saved_cfg)

    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()

    if hasattr(task, "tokenizer"):
        task.tokenizer = task.build_tokenizer(cfg.tokenizer)
    if hasattr(task, "bpe"):
        task.bpe = task.build_bpe(cfg.bpe)

    for subset in cfg.dataset.gen_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1, task_cfg=saved_cfg.task)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        )

        predictions = []
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            preds = task.inference(task.sequence_generator, sample, model, for_submit=True)
            progress.log({}, step=i)
            predictions.extend(preds)

        with open("predictions.tsv", "w", encoding="utf-8") as writer:
            writer.write("index\tprediction\n")
            for i, prediction in enumerate(predictions):
                writer.write(f"{i}\t{prediction}\n")


def cli_main():
    try:
        from hydra._internal.utils import get_args

        cfg_name = get_args().config_name or "config"
    except:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "config"

    hydra_init(cfg_name)
    hydra_main()


@hydra.main(config_path=os.path.join("..", "..", "fairseq", "config"), config_name="config")
def hydra_main(cfg: FairseqConfig) -> float:
    _hydra_main(cfg)


def _hydra_main(cfg: FairseqConfig, **kwargs) -> float:
    add_defaults(cfg)

    if cfg.common.reset_logging:
        reset_logging()  # Hydra hijacks logging, fix that
    else:
        # check if directly called or called through hydra_main
        if HydraConfig.initialized():
            with open_dict(cfg):
                # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
                cfg.job_logging_cfg = OmegaConf.to_container(
                    HydraConfig.get().job_logging, resolve=True
                )

    with omegaconf_no_object_check():
        cfg = OmegaConf.create(
            OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        )
    OmegaConf.set_struct(cfg, True)

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main, **kwargs)
        else:
            distributed_utils.call_main(cfg, main, **kwargs)
    except BaseException as e:
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! " + str(e))


if __name__ == "__main__":
    cli_main()
