from .utilities3 import count_params
from .Transolver import Transolver, Transolver_Cd
from .Transolver_conv_proj import Transolver_conv_proj
from .RegDGCNN import RegDGCNN


def instantiate_network(config):
    out_channels = config.out_channels  # pressure
    print(config.model)
    if config.model == "Transolver":
        print("using Transolver")
        model = Transolver(
            n_hidden=config.n_hidden,
            n_layers=config.n_layers,
            space_dim=config.space_dim,
            fun_dim=config.fun_dim,
            n_head=config.n_head,
            mlp_ratio=config.mlp_ratio,
            out_dim=config.out_dim,
            slice_num=config.slice_num,
            unified_pos=config.unified_pos,
            subsample_train=config.subsample_train,
            subsample_eval=config.subsample_eval,
        )
    elif config.model == "Transolver_Cd":
        print("using Transolver_Cd")
        model = Transolver_Cd(
            n_hidden=config.n_hidden,
            n_layers=config.n_layers,
            space_dim=config.space_dim,
            fun_dim=config.fun_dim,
            n_head=config.n_head,
            mlp_ratio=config.mlp_ratio,
            out_dim=config.out_dim,
            slice_num=config.slice_num,
            unified_pos=config.unified_pos,
            subsample_train=config.subsample_train,
            subsample_eval=config.subsample_eval,
        )
    elif config.model == "Transolver_conv_proj":
        print("using Transolver_conv_proj")
        model = Transolver_conv_proj(
            n_hidden=config.n_hidden,
            n_layers=config.n_layers,
            space_dim=config.space_dim,
            fun_dim=config.fun_dim,
            n_head=config.n_head,
            mlp_ratio=config.mlp_ratio,
            out_dim=config.out_dim,
            slice_num=config.slice_num,
            unified_pos=config.unified_pos,
            subsample_train=config.subsample_train,
            subsample_eval=config.subsample_eval,
        )
    elif config.model == "RegDGCNN":
        print("using RegDGCNN")
        model = RegDGCNN(
            args=config,
        )
    else:
        raise ValueError("Network not supported")

    # print(model)  # 这一句显示网络结构
    print("The model size is ", count_params(model))
    return model
