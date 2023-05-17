import torch

from transformers.optimization import AdamW
from transformers import (
    get_constant_schedule,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def set_scheduler(pl_module):
    lr = pl_module.config["learning_rate"]
    wd = pl_module.config["weight_decay"]
    optim_type = pl_module.config["optim_type"]
    decay_power = pl_module.config["decay_power"]
    end_lr = pl_module.config["end_lr"]
    warmup_steps = pl_module.config["warmup_steps"]

    max_steps = pl_module.trainer.max_steps

    # Define Optimizer
    if optim_type == "adamw":
        optimizer = AdamW(pl_module.parameters(), lr=lr, eps=1e-8, betas=(0.9, 0.98))
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(pl_module.parameters(), lr=lr)
    elif optim_type == "adam_diff":
        params = list(pl_module.named_parameters())

        def is_backbone(n):
            return "syncnet" in n

        grouped_parameters = [
            {
                "params": [p for n, p in params if is_backbone(n)],
                "lr": lr * 0.0000001,
            },
            {
                "params": [p for n, p in params if not is_backbone(n)],
                "lr": lr,
            },
        ]
        optimizer = torch.optim.Adam(grouped_parameters, lr=lr)

    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(
            pl_module.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5
        )

    # Define Learning Scheduler
    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
        )
    elif decay_power == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
        )
    elif decay_power == "constant":
        scheduler = get_constant_schedule(optimizer)
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}
    return ([optimizer], [sched])
