
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

import os
import copy

from config import ex
from model.face_tts import FaceTTS
from data import _datamodules

@ex.automain
def main(_config):

    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = _datamodules["dataset_" + _config["dataset"]](_config)

    os.makedirs(_config["local_checkpoint_dir"], exist_ok=True)

    checkpoint_callback_epoch = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/total_loss",
        mode="min",
        save_last=True,
        auto_insert_metric_name=True,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    model = FaceTTS(_config)

    model_summary_callback = pl.callbacks.ModelSummary(max_depth=2)

    callbacks = [checkpoint_callback_epoch, lr_callback, model_summary_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        strategy=DDPPlugin(gradient_as_bucket_view=True, find_unused_parameters=True),
        max_steps=max_steps,
        callbacks=callbacks,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=50,
        flush_logs_every_n_steps=50,
        weights_summary="top",
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm, ckpt_path=_config["resume_from"])
    else:
        trainer.test(model, datamodule=dm, ckpt_path=_config["resume_from"])
