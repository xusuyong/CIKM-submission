import matplotlib

matplotlib.use("Agg")  # Set the backend to Agg

import os
import csv
import pandas as pd
from typing import List, Tuple, Union
import numpy as np
import yaml
from timeit import default_timer
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import torch

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


def parse_args(yaml="UnetShapeNetCar.yaml"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=yaml,
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

    args, _ = parser.parse_known_args()
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
        out_dict, _ = model.eval_dict(
            data_dict, loss_fn=loss_fn, decode_fn=None
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


def train(config, args, device: Union[torch.device, str] = "cuda:0"):
    # Initialize the device
    device = torch.device(device)
    loggers, log_dir = init_logger(config)
    config.log_dir=log_dir
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
    Transolver_type_model = ["Transolver", "Transolver_conv_proj", "Transolver-Cd"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=0 if config.model in Transolver_type_model else 1e-4,
    )
    scheduler = instantiate_scheduler(optimizer, config)

    # Initialize the loss function
    loss_fn = LpLoss(size_average=True)
    # loss_fn = torch.nn.MSELoss(reduction='mean')

    # N_sample = 1000
    for ep in range(config.num_epochs):
        model.train()
        t1 = default_timer()
        train_l2_meter = AverageMeter()
        # train_reg = 0
        # ['vertices'],([1, 3586, 3]). ['pressure'] ([1, 3586])
        for data_dict in train_loader:
            optimizer.zero_grad()
            loss_dict = model.loss_dict(data_dict, loss_fn=loss_fn)
            loss = 0
            for k, v in loss_dict.items():
                loss = loss + v.mean()
            loss.backward()

            optimizer.step()

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
            # print(f"saving model to ./{log_dir}/model-{config.model}-{ep}.pt")
            # torch.save(model, os.path.join(f"./{log_dir}/", f"model-{config.model}-{config.track}-{ep}.pt"))
            print(f"saving model state to ./{log_dir}/model-{config.model}-{config.track}-{ep}.pth")
            torch.save(
                model.state_dict(),
                os.path.join(f"./{log_dir}/", f"model_state-{config.model}-{config.track}-{ep}.pth"),
            )

def load_yaml(file_name):
    args = parse_args(file_name)
    # args = parse_args("Unet_Velocity.yaml")
    config = load_config(args.config)

    # Update config with command line arguments
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value

    # pretty print the config
    if True:
        print(f"\n--------------- Config [{file_name}] Table----------------")
        for key, value in config.items():
            print("Key: {:<30} Val: {}".format(key, value))
        print("--------------- Config yaml Table----------------\n")
    return config, args

import re
def extract_numbers(s):
    return [int(digit) for digit in re.findall(r'\d+', s)]

def write_to_vtk(out_dict, point_data_pos="press on mesh points", mesh_path=None, track=None):
    import meshio
    p = out_dict["pressure"]
    index = extract_numbers(mesh_path.name)[0]

    if track == "Dataset_1":
        index = str(index).zfill(3)   
    elif track == "Track_B":
        index = str(index).zfill(4)

    print(f"Pressure shape for mesh {index} = {p.shape}")

        
    if point_data_pos == "press on mesh points":
        mesh = meshio.read(mesh_path)
        mesh.point_data["p"] = p.numpy()
        if "pred wss_x" in out_dict:
            wss_x = out_dict["pred wss_x"]
            mesh.point_data["wss_x"] = wss_x.numpy()
    elif point_data_pos == "press on mesh cells":
        points = np.load(mesh_path.parent / f"centroid_{index}.npy")
        npoint = points.shape[0]
        mesh = meshio.Mesh(
            points=points, cells=[("vertex", np.arange(npoint).reshape(npoint, 1))]
        )
        mesh.point_data = {"p":p.numpy()}

    print(f"write : ./output/{mesh_path.parent.name}_{index}.vtk")
    mesh.write(f"./output/{mesh_path.parent.name}_{index}.vtk") 

@torch.no_grad()
def infer(model, datamodule, config, loss_fn=None, track="Dataset_1"):
    model.eval()
    test_loader = datamodule.test_dataloader(batch_size=config.eval_batch_size, shuffle=False, num_workers=0)
    data_list = []
    cd_list = []
    global_index = 0
    for i, data_dict in enumerate(test_loader):
        out_loss_dict, output_dict = model.eval_dict(data_dict, loss_fn=loss_fn, decode_fn=datamodule.decode)
        if'l2 eval loss' in out_loss_dict: 
            if i == 0:
                data_list.append(['id', 'l2 p'])
            else:
                data_list.append([i, float(out_loss_dict['l2 eval loss'])])
        
        # TODO : you may write velocity into vtk, and analysis in your report
        if config.write_to_vtk is True:
            print("datamodule.test_mesh_paths = ", datamodule.test_mesh_paths[i])
            write_to_vtk(output_dict, config.point_data_pos, datamodule.test_mesh_paths[i], track)
        
        # Your submit your npy to leaderboard here
        if "pressure" in output_dict:
            p = output_dict["pressure"].reshape((-1,)).astype(np.float32)
            test_indice = datamodule.test_indices[i]
            npy_leaderboard = f"./output/{track}/press_{str(test_indice).zfill(3)}.npy"
            print(f"saving *.npy file for [{track}] leaderboard : ", npy_leaderboard)
            np.save(npy_leaderboard, p)
        if "velocity" in output_dict:
            v = output_dict["velocity"].cpu().reshape((-1,3)).numpy()
            test_indice = datamodule.test_indices[i]
            npy_leaderboard = f"./output/{track}/vel_{str(test_indice).zfill(3)}.npy"
            print(f"saving *.npy file for [{track}] leaderboard : ", npy_leaderboard)
            np.save(npy_leaderboard, v)
        if "cd" in output_dict:
            cd_values = output_dict["cd"].squeeze(1)  
            # 假设你有一个样本索引列表（这里我们使用range来模拟）  
            # 注意：在实际应用中，这些索引可能来自你的数据加载器或数据模块  
            batch_size = cd_values.size(0)  
            for i in range(batch_size):  
                # 使用global_index来确保索引在整个数据集中是连续的  
                cd_list.append([global_index, cd_values[i].item()])  
                global_index += 1  
            # v = output_dict["cd"].item()
            # test_indice = datamodule.test_indices[i]
            # cd_list.append([i, v])

        # check csv in ./output
        with open(f"./output/{config.project_name}.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data_list)

    if "cd" in output_dict:
        titles = ["", "Cd"]
        df = pd.DataFrame(cd_list, columns=titles)
        df.to_csv(f'./output/{track}/Answer.csv', index=False)
    return

def leader_board(config, track):
    os.makedirs(f"./output/{track}/", exist_ok=True)
    
    model = instantiate_network(config)  # 实例化网络
    checkpoint = torch.load(f"{config.model_path}")
    model.load_state_dict(checkpoint)
    # model.load_state_dict(torch.load("/home/xusuyong/pythoncode/myproj/CIKM-submission/logs/2024-09-15_09-49-42/model_state-Transolver_conv_proj-Dataset_1_velocity-249.pth"))
    

    # model = instantiate_network(config)
    # model = torch.load(f"{config.log_dir}/model-{config.model}-{config.track}-{config.num_epochs - 1}.pt")
    # model = torch.load(f"{config.model_path}")
    # torch.save(model, os.path.join(f"./", f"model-{config.model}-{config.track}.pt"))
    # torch.save(model.state_dict(), os.path.join(f"./", f"model-{config.model}-{config.track}.pth"))
    exit()
    # checkpoint = torch.load(f"{config.log_dir}/model_state-{config.model}-{config.track}-{config.num_epochs - 1}.pth")
    # model.load_state_dict(checkpoint)
    print(f"\n-------Starting Evaluation over [{config.track}] --------")
    config.n_train = 1
    t1 = default_timer()
    
    config.mode="test"
    datamodule = instantiate_datamodule(config)
    eval_dict = infer(model, datamodule, config, loss_fn=lambda x,y:0, track=track)
    t2 = default_timer()
    print(f"Inference over [Dataset_1 pressure] took {t2 - t1:.2f} seconds.")


if __name__ == "__main__":
    os.makedirs("./output/", exist_ok=True)
    config_p, args = load_yaml("configs/transolver/Conv-Transolver_Press.yaml")
    config_p.n_test = 1
    # train(config_p, args)

    index_list = np.loadtxt("../../xsy_datasets/CIKM_dataset/AIstudio_CIKM_dataset/train_data_1_velocity/watertight_meshes.txt", dtype=int)
    config_v, args = load_yaml("configs/transolver/Conv-Transolver_Velocity.yaml")
    config_v.train_index_list = index_list[: config_v.n_train].tolist()
    config_v.test_index_list = index_list[500 : 500 + config_v.n_test].tolist()
    # train(config_v, args)
    # exit()

    config_cd, args = load_yaml("configs/transolver/RegDGCNN_Cd.yaml")
    index_list = np.loadtxt("../../xsy_datasets/CIKM_dataset/AIstudio_CIKM_dataset/Training/Dataset_2/Label_File/dataset2_train_label.csv", delimiter=",", dtype=str, encoding='utf-8')[:,1][1:]
    config_cd.train_index_list = index_list[:500].tolist()
    config_cd.test_index_list = index_list[500:550].tolist()
    # train(config_cd, args)

    # test on leader_board, or do evaluation by yourself
    leader_board(config_p,  "Gen_Answer")
    leader_board(config_v,  "Gen_Answer")
    # leader_board(config_cd, "Gen_Answer")
    # os.system(f"zip -r -j ./output/Gen_Answer.zip ./output/Gen_Answer")
