import matplotlib

matplotlib.use("Agg")  # Set the backend to Agg

import os
from typing import List, Tuple, Union
import numpy as np
import yaml
from timeit import default_timer
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data import instantiate_datamodule
from src.networks import instantiate_network
from src.utils.average_meter import AverageMeter, AverageMeterDict
from src.utils.dot_dict import DotDict, flatten_dict
from src.losses import LpLoss
from src.utils.loggers import init_logger
from src.optim.schedulers import instantiate_scheduler


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def str2intlist(s: str) -> List[int]:
    return [int(item.strip()) for item in s.split(",")]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="my_configs/trackB/UNetAhmed.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (cuda or cpu)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../xsy_datasets/GINO_dataset/car-pressure-data",
        help="Override data_path in config file",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file to resume training",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="log",
        help="Path to the log directory",
    )
    parser.add_argument("--logger_types", type=str, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for training")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument(
        "--sdf_spatial_resolution",
        type=str2intlist,
        default=None,
        help="SDF spatial resolution. Use comma to separate the values e.g. 32,32,32.",
    )
    parser.add_argument(
        "--world_size", type=int, default=2, help="Number of processes for DDP"
    )
    parser.add_argument(
        "--rank", type=int, default=0, help="Rank of the process for DDP"
    )
    parser.add_argument(
        "--dist_url",
        type=str,
        default="tcp://127.0.0.1:23456",
        help="URL used to set up distributed training",
    )

    args = parser.parse_args()
    return args


def load_config(config_path):
    def include_constructor(loader, node):
        current_file_path = loader.name
        base_folder = os.path.dirname(current_file_path)
        included_file = os.path.join(base_folder, loader.construct_scalar(node))
        with open(included_file, "r") as file:
            return yaml.load(file, Loader=yaml.Loader)

    yaml.Loader.add_constructor("!include", include_constructor)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config_flat = flatten_dict(config)
    config_flat = DotDict(config_flat)
    return config_flat


@torch.no_grad()
def eval(model, datamodule, config, loss_fn=None):
    model.eval()
    test_loader = datamodule.test_dataloader(
        batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    eval_meter = AverageMeterDict()
    visualize_data_dicts = []
    for i, data_dict in enumerate(test_loader):
        out_dict = model.module.eval_dict(
            data_dict, loss_fn=loss_fn, decode_fn=None
        )
        eval_meter.update(out_dict)
        if i % config.test_plot_interval == 0:
            visualize_data_dicts.append(data_dict)

    merged_image_dict = {}
    if hasattr(model.module, "image_dict"):
        for i, data_dict in enumerate(visualize_data_dicts):
            image_dict = model.module.image_dict(data_dict)
            for k, v in image_dict.items():
                merged_image_dict[f"{k}_{i}"] = v

    model.train()

    return eval_meter.avg, merged_image_dict


def train(rank, world_size):
    args = parse_args()
    config = load_config(args.config)

    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value

    if config.seed is not None:
        set_seed(config.seed)

    config.world_size=world_size

    dist.init_process_group(
        backend="nccl", init_method=config.dist_url, world_size=world_size, rank=rank
    )
    torch.cuda.set_device(rank)
    
    if rank == 0:
        loggers, log_dir = init_logger(config)
        os.system(f"cp {args.config} {log_dir}")
        
    # Initialize the model
    if config.pretrained_model:
        print("-" * 10 + "loading pretrained model" + "-" * 10)
        model = torch.load(config.pretrained_model_path).to(rank)
    else:
        model = instantiate_network(config).to(rank)  # 实例化网络

    model = DDP(model, device_ids=[rank])

    datamodule = instantiate_datamodule(config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        datamodule.train_data, num_replicas=world_size, rank=rank
    )
    train_loader = datamodule.train_dataloader(
        batch_size=config.batch_size, shuffle=False, sampler=train_sampler
    )

    Transolver_type_model = ["Transolver", "Transolver_conv_proj"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=0 if config.model in Transolver_type_model else 1e-4,
    )
    scheduler = instantiate_scheduler(optimizer, config)

    loss_fn = LpLoss(size_average=True)

    for ep in range(config.num_epochs):
        model.train()
        train_sampler.set_epoch(ep)
        t1 = default_timer()
        train_l2_meter = AverageMeter()

        for data_dict in train_loader:
            optimizer.zero_grad()
            loss_dict = model.module.loss_dict(data_dict, loss_fn=loss_fn)
            loss = 0
            for k, v in loss_dict.items():
                loss = loss + v.mean()
            loss.backward()
            optimizer.step()

            train_l2_meter.update(loss.item())

            if rank == 0:
                loggers.log_scalar("train/lr", scheduler.get_last_lr()[0], ep)
                loggers.log_scalar("train/loss", loss.item(), ep)

            if config.opt_scheduler == "OneCycleLR":
                scheduler.step()

        if config.opt_scheduler != "OneCycleLR":
            scheduler.step()

        t2 = default_timer()
        if rank == 0:
            print(
                f"Training epoch {ep} took {t2 - t1:.2f} seconds. L2 loss: {train_l2_meter.avg:.6f}"
            )
            loggers.log_scalar("train/train_l2", train_l2_meter.avg, ep)
            loggers.log_scalar("train/train_epoch_duration", t2 - t1, ep)

        if (ep % config.eval_interval == 0 or ep == config.num_epochs - 1) and rank == 0:
            eval_dict, eval_images = eval(model, datamodule, config, loss_fn)
            for k, v in eval_dict.items():
                print(f"Epoch: {ep} {k}: {v:.4f}")
                loggers.log_scalar(f"eval/{k}", v, ep)
            for k, v in eval_images.items():
                loggers.log_image(f"eval/{k}", v, ep)

        # Save the weights
        if (ep % config.eval_interval == 0 or ep == config.num_epochs - 1) and rank == 0:
            print(f"saving model_module to ./{log_dir}/model_module-{config.model}-{ep}.pt")
            torch.save(model.module, os.path.join(f"./{log_dir}/", f"model_module-{config.model}-{ep}.pt"))
            # print(f"saving model_module state to ./{log_dir}/model-{config.model}-{ep}.pth")
            # torch.save(model.module.state_dict(), os.path.join(f"./{log_dir}/", f"model_module_state_dict-{config.model}-{ep}.pth"))

    dist.destroy_process_group()


if __name__ == "__main__":

    world_size = torch.cuda.device_count()
    # world_size=2
    if world_size > 1:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)#world_size之后一定要有逗号
    else:
        train(0, 1)
