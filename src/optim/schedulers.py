import torch


def instantiate_scheduler(optimizer, config):
    if config.opt_scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.opt_scheduler_T_max
        )
    elif config.opt_scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.opt_step_size, gamma=config.opt_gamma
        )
    elif config.opt_scheduler == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.lr,
            total_steps=(
                (config.n_train // (config.world_size if config.world_size else 1))
                // config.batch_size
                + 1
            )
            * config.num_epochs,
            final_div_factor=1000.0,
        )
    else:
        raise ValueError(f"Got {config.opt.scheduler=}")
    return scheduler
