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
from torch.cuda.amp import autocast, GradScaler
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

    args = parser.parse_args()
    return args


def load_config(config_path):
    def include_constructor(loader, node):
        # Get the path of the current YAML file
        current_file_path = loader.name

        # Get the folder containing the current YAML file
        base_folder = os.path.dirname(current_file_path)

        # Get the included file path, relative to the current file
        included_file = os.path.join(base_folder, loader.construct_scalar(node))

        # Read and parse the included file
        with open(included_file, "r") as file:
            return yaml.load(file, Loader=yaml.Loader)

    # Register the custom constructor for !include
    yaml.Loader.add_constructor("!include", include_constructor)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Convert to dot dict
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
        with autocast():
            out_dict = model.eval_dict(
                data_dict, loss_fn=loss_fn, decode_fn=datamodule.decode
            )
        eval_meter.update(out_dict)
        if i % config.test_plot_interval == 0:
            visualize_data_dicts.append(data_dict)

    # Merge all dictionaries
    merged_image_dict = {}
    if hasattr(model, "image_dict"):
        for i, data_dict in enumerate(visualize_data_dicts):
            image_dict = model.image_dict(data_dict)
            for k, v in image_dict.items():
                merged_image_dict[f"{k}_{i}"] = v

    model.train()

    return eval_meter.avg, merged_image_dict


def train(config, device: Union[torch.device, str] = "cuda:0"):
    # Initialize the device
    device = torch.device(device)
    loggers, log_dir = init_logger(config)
    # 将配置文件复制到日志文件夹中
    os.system(f"cp {args.config} {log_dir}")
    # Initialize the model
    if config.pretrained_model:
        print("-" * 15 + "loading pretrained model" + "-" * 15)
        model = torch.load(config.pretrained_model_path).to(device)
    else:
        model = instantiate_network(config).to(device)  # 实例化网络
    # Initialize the dataloaders
    datamodule = instantiate_datamodule(config)
    train_loader = datamodule.train_dataloader(
        batch_size=config.batch_size, shuffle=True, num_workers=8
    )

    # Initialize the optimizer
    Transolver_type_model = ["Transolver", "Transolver_conv_proj"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=0 if config.model in Transolver_type_model else 1e-4,
    )
    scheduler = instantiate_scheduler(optimizer, config)

    # Initialize the loss function
    loss_fn = LpLoss(size_average=True)
    # loss_fn = torch.nn.MSELoss(reduction='mean')
    scaler = GradScaler()
    
    # N_sample = 1000
    for ep in range(config.num_epochs):
        model.train()
        t1 = default_timer()
        train_l2_meter = AverageMeter()
        # train_reg = 0
        # ['vertices'],([1, 3586, 3]). ['pressure'] ([1, 3586])
        for data_dict in train_loader:
            optimizer.zero_grad()
            
            with autocast():
                loss_dict = model.loss_dict(data_dict, loss_fn=loss_fn)
                loss = 0
                for k, v in loss_dict.items():
                    loss = loss + v.mean()
                # loss.backward()
                # 反向传播和优化
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # optimizer.step()

            train_l2_meter.update(loss.item())

            loggers.log_scalar("train/lr", scheduler.get_last_lr()[0], ep)
            loggers.log_scalar("train/loss", loss.item(), ep)
            """Transolver更新在这里！！！"""
            if config.opt_scheduler == "OneCycleLR":
                scheduler.step()
        if config.opt_scheduler != "OneCycleLR":
            scheduler.step()
        t2 = default_timer()
        print(
            f"Training epoch {ep} took {t2 - t1:.2f} seconds. L2 loss: {train_l2_meter.avg:.4f}"
        )
        loggers.log_scalar("train/train_l2", train_l2_meter.avg, ep)
        loggers.log_scalar("train/train_epoch_duration", t2 - t1, ep)

        if ep % config.eval_interval == 0 or ep == config.num_epochs - 1:
            eval_dict, eval_images = eval(model, datamodule, config, loss_fn)
            for k, v in eval_dict.items():
                print(f"Epoch: {ep} {k}: {v:.4f}")
                loggers.log_scalar(f"eval/{k}", v, ep)
            for k, v in eval_images.items():
                loggers.log_image(f"eval/{k}", v, ep)

        # Save the weights
        if ep % config.eval_interval == 0 or ep == config.num_epochs - 1:
            print(f"saving model to ./{log_dir}/model-{config.model}-{ep}.pt")
            torch.save(model, os.path.join(f"./{log_dir}/", f"model-{config.model}-{ep}.pt"))
            print(f"saving model state to ./{log_dir}/model-{config.model}-{ep}.pth")
            torch.save(
                model.state_dict(),
                os.path.join(f"./{log_dir}/", f"model_state-{config.model}-{ep}.pth"),
            )


if __name__ == "__main__":
    args = parse_args()
    # print command line args
    print(args)
    config = load_config(args.config)

    # Update config with command line arguments
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value

    # pretty print the config
    for key, value in config.items():
        print(f"{key}: {value}")

    # Set the random seed
    if config.seed is not None:
        set_seed(config.seed)
    train(config, device=args.device)
