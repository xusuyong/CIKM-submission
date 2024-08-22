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
import csv
import torch

from src.data import instantiate_datamodule
from src.networks import instantiate_network
from src.utils.average_meter import AverageMeter, AverageMeterDict
from src.utils.dot_dict import DotDict, flatten_dict
from src.losses import LpLoss
from src.utils.loggers import init_logger
from src.optim.schedulers import instantiate_scheduler
import re


def str2intlist(s: str) -> List[int]:
    return [int(item.strip()) for item in s.split(",")]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="my_configs/trackB/TransolverTrackB_conv_proj.yaml",
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
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
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


def extract_numbers(s):
    return [int(digit) for digit in re.findall(r"\d+", s)]


def write_to_vtk(out_dict, point_data_pos="press on mesh points", mesh_path=None):
    import meshio

    p = out_dict["pressure"]
    index = extract_numbers(mesh_path.name)[0]
    index = str(index).zfill(3)

    if point_data_pos == "press on mesh points":  # 执行这句
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
        mesh.point_data = {"p": p.numpy()}

    os.makedirs("./output/gen_answer_A/content/gen_answer_A/", exist_ok=True)
    os.makedirs("./output/visualize", exist_ok=True)
    print(f"write : ./output/visualize/{mesh_path.parent.name}_{index}.vtk")
    mesh.write(f"./output/visualize/{mesh_path.parent.name}_{index}.vtk")
    np.save(
        f"./output/gen_answer_A/content/gen_answer_A/press_{index}.npy",
        p.numpy().squeeze().astype(np.float64),
    )


def save_npy(
    out_dict,
    i,
    point_data_pos="press on mesh points",
):
    # Your submit your npy to leaderboard here
    p = out_dict["pressure"].squeeze()

    track = "Track_B"
    os.makedirs(f"./output/{track}", exist_ok=True)
    os.makedirs(f"{modeldir.parent}/{track}", exist_ok=True)
    n = 0
    test_indice = datamodule.test_indices[i]
    # npy_leaderboard = f"./output/{track}/press_{str(test_indice).zfill(n)}.npy"
    npy_leaderboard = f"{modeldir.parent}/{track}/press_{str(test_indice).zfill(n)}.npy"
    print(f"saving *.npy file for [{track}] leaderboard : ", npy_leaderboard)
    np.save(npy_leaderboard, p)
    # np.save(modeldir.parent, p)
    # import meshio

    # p = out_dict["pressure"]
    # # index = extract_numbers(mesh_path.name)[0]
    # index = str(index).zfill(3)

    # os.makedirs("./output/visualize", exist_ok=True)

    # np.save(f"./output/gen_answer_A/content/gen_answer_A/press_{index}.npy", p.numpy().squeeze().astype(np.float64))


@torch.no_grad()
def infer(model, datamodule, config, loss_fn=None):
    model.eval()
    # infer_loader = datamodule.infer_dataloader(
    #     batch_size=1, shuffle=False, num_workers=0
    # )  # 注意这个batchsize一定要指定
    test_loader = datamodule.test_dataloader(
        batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    data_list = []
    averaged_output_dict = {}
    os.makedirs("./output/", exist_ok=True)

    for i, data_dict in enumerate(test_loader):
        class_name = type(model).__name__  
        print("infer model:", class_name)

        if class_name == "GNO":
            vert = model.data_dict_to_input(data_dict)  # torch.Size([1, 3586, 3])
            press = model(vert).detach().cpu()  # torch.Size([1, 3586, 1])
        elif class_name == "GNOFNOGNO":
            x_in, x_out, df = model.data_dict_to_input(data_dict)
            pred = model(x_in, x_out, df).reshape(1, -1).detach().cpu()
            press = datamodule.decode(pred)
            # press = pred
        elif class_name == "GNOFNOGNOTrackB":
            x_in, x_out, df, area = model.data_dict_to_input(data_dict)
            pred = model(x_in=x_in, x_out=x_out, df=df, x_eval=None, area_in=area, area_eval=None).reshape(1, -1).detach().cpu()
            press = datamodule.decode(pred)
        # elif class_name == "GNOFNOGNOTrackB":
        #     x_in, x_out, df, area = model.data_dict_to_input(data_dict)

        #     sampled_x_in = [x_in[i::config.subsample_eval] for i in range(config.subsample_eval)]
        #     sampled_areas = [area[i::config.subsample_eval] for i in range(config.subsample_eval)]

        #     final_output = torch.zeros(area.shape[0])
        #     print(final_output.shape)
        #     for idx in range(len(sampled_x_in)):
        #         pred_mini = model(x_in=sampled_x_in[idx], x_out=x_out, df=df, x_eval=None, area_in=sampled_areas[idx], area_eval=None).squeeze().detach().cpu()
        #         print(pred_mini.shape)
        #         final_output[idx::config.subsample_eval] = pred_mini
        #     pred = final_output
        #     print(pred.shape)
        #     press = datamodule.decode(pred)
        elif class_name in ["Transolver", "Transolver_conv_proj"]:
            vert = model.data_dict_to_input(data_dict)
            pred = model(vert).reshape(1, -1).detach().cpu()
            # press = datamodule.decode(pred)
            press = pred
        # elif class_name == "Transolver":
        #     vert = model.data_dict_to_input(data_dict)

        #     sampled_vert = [vert[i::config.subsample_eval] for i in range(config.subsample_eval)]

        #     final_output = torch.zeros(vert.shape[0])
        #     print(final_output.shape)
        #     for idx in range(len(sampled_vert)):
        #         pred_mini = model(sampled_vert[idx]).squeeze().detach().cpu()
        #         print(pred_mini.shape)
        #         final_output[idx::config.subsample_eval] = pred_mini
        #     pred = final_output
        #     print(pred.shape)
        #     press = datamodule.decode(pred)

        out_dict = {"pressure": press.view(-1, 1)}
        # out_dict = model.eval_dict(
        #     data_dict, loss_fn=loss_fn, decode_fn=datamodule.decode
        # )

        if config.write_to_vtk is True:
            # print("datamodule.test_mesh_paths = ", datamodule.test_mesh_pathes[i])
            # write_to_vtk(
            #     out_dict, "press on mesh points", datamodule.test_mesh_pathes[i]
            # )
            save_npy(out_dict, i)

        with open(f"./output/{config.project_name}.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data_list)

    return averaged_output_dict


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


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    from pathlib import Path

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
    print("\n-------Starting Evaluation over [track B]--------")
    # Initialize the device
    device = torch.device("cuda")
    # device = torch.device("cpu")

    # Initialize the model
    # model = instantiate_network(config).to(device)  # 实例化网络
    # model.load_state_dict(torch.load("logs/model-Transolver-0.pth"))
    if config.model_path == None:
        modeldir = Path("logs/2024-07-02_15-27-44/model_module-Transolver_conv_proj-799.pt")
    else:
        modeldir = Path(config.model_path)

    model = torch.load(modeldir).to(device)

    t1 = default_timer()
    datamodule = instantiate_datamodule(config)
    eval_dict = infer(model, datamodule, config, loss_fn=LpLoss(size_average=True))
    t2 = default_timer()
    print(f"Inference over [track B] took {t2 - t1:.2f} seconds.")

    os.system(f"zip -r {modeldir.parent}/Track_B.zip {modeldir.parent}/Track_B")
