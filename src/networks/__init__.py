# from .GNO import GNO
# from .GeoFNO import GeoFNO
# from .GeoFNO2d import GeoFNO2d
# from .ContinuousConvNet import SmallContConvWithMLPKernel
# from .ConvUNet2 import UNet3DWithSamplePoints, UNet3DWithSamplePointsAhmed
# from .GNOFNOGNO import GNOFNOGNO, GNOFNOGNOAhmed, GNOFNOGNOTrackB
# from .FNOInterp import FNOInterp, FNOInterpAhmed
# from .FNOGNO import FNOGNO, FNOGNOAhmed
from .utilities3 import count_params
# from .SDFFNOGNO import SDFFNOGNO
# from .Transolver import Transolver
from .Transolver_conv_proj import Transolver_conv_proj
# from .Transolver_conv_multifi import Transolver_conv_multifi
# from .Transolver_conv_sdf import Transolver_conv_sdf

from .TransGINO import TransGINO


def instantiate_network(config):
    out_channels = 1  # pressure
    print(config.model)

    if config.model == "GeoFNO":
        print("using GeoFNO3d")
        model = GeoFNO(
            config.modes,
            config.modes,
            config.modes,
            config.width,
            out_channels=out_channels,
            s=config.s,
        )
    elif config.model == "GeoFNO2d":
        print("using GeoFNO2d")
        model = GeoFNO2d(
            config.modes,
            config.modes,
            config.modes,
            config.width,
            out_channels=out_channels,
            s=config.s,
        )
    elif config.model == "GNO":
        print("using GNO")
        model = GNO(width=config.width, out_channel=out_channels, r=config.r)
    elif config.model == "SDFFNO":
        print("using SDFFNO")
        model = SDFFNO(
            n_modes=config.fno_modes,
            hidden_channels=config.hidden_channels,
            norm=config.norm,
            use_mlp=config.use_mlp,
            mlp={"expansion": 1.0, "dropout": 0},
            domain_padding=config.domain_padding,
            factorization=config.factorization,
            rank=0.4,
            out_channels=out_channels,
        )
    elif config.model == "SDFFNOGNO":
        print("using SDFFNOGNO")
        model = SDFFNOGNO(
            n_modes=config.fno_modes,
            hidden_channels=config.hidden_channels,
            norm=config.norm,
            use_mlp=config.use_mlp,
            mlp={"expansion": 1.0, "dropout": 0},
            domain_padding=config.domain_padding,
            factorization=config.factorization,
            rank=0.4,
            out_channels=out_channels,
            r=config.r,
            resolution=config.sdf_spatial_resolution,
            gno_implementation="torch_scatter",
            subsample_train=config.subsample_train,
            subsample_eval=config.subsample_eval,
        )
    elif config.model == "SmallContConvWithMLPKernel":
        model = SmallContConvWithMLPKernel(
            out_channel=out_channels, width=config.width, radius=config.r
        )
    elif config.model == "UNet":
        in_channels = 4 if config.use_position_input else 1
        model = UNet3DWithSamplePoints(
            in_channels=in_channels,  # xyz + sdf
            out_channels=out_channels,
            hidden_channels=config.hidden_channels,
            num_levels=config.num_levels,
            use_position_input=config.use_position_input,
        )
    elif config.model == "UNetAhmed":
        model = UNet3DWithSamplePointsAhmed(
            in_channels=5,  # xyz + sdf + vel
            out_channels=out_channels,
            hidden_channels=config.hidden_channels,
            num_levels=config.num_levels,
            use_position_input=config.use_position_input,
            subsample_train=config.subsample_train,
            subsample_eval=config.subsample_eval,
        )
    elif config.model == "FNOInterp":
        print("using FNOInterp")
        model = FNOInterp()
    elif config.model == "FNOInterpAhmed":
        print("using FNOInterpAhmed")
        model = FNOInterpAhmed(
            in_channels=12,
            out_channels=1,
            fno_modes=(32, 32, 32),
            fno_hidden_channels=64,
            fno_domain_padding=0.125,
            fno_norm="ada_in",
            fno_factorization="tucker",
            fno_rank=0.4,
            embed_dim=256,
            subsample_train=config.subsample_train,
            subsample_eval=config.subsample_eval,
        )
    elif config.model == "GNOFNOGNO":
        print("using GNOFNOGNO")
        model = GNOFNOGNO(
            radius_in=config.radius_in,
            radius_out=config.radius_out,
            embed_dim=32,
            hidden_channels=(64, 64),
            in_channels=1,
            out_channels=1,
            fno_modes=(32, 32, 32),
            fno_hidden_channels=64,
            fno_out_channels=64,
            fno_domain_padding=0.125,
            fno_norm="group_norm",
            fno_factorization="tucker",
            fno_rank=0.4,
        )
    elif config.model == "GNOFNOGNOAhmed":
        print("using GNOFNOGNOAhmed")
        model = GNOFNOGNOAhmed(
            radius_in=0.035,
            radius_out=0.035,
            embed_dim=32,
            hidden_channels=(64, 64),
            in_channels=2,
            out_channels=1,
            fno_modes=(32, 32, 32),
            fno_hidden_channels=64,
            fno_out_channels=64,
            fno_domain_padding=0.125,
            fno_norm="ada_in",
            fno_factorization="tucker",
            fno_rank=0.4,
            linear_kernel=True,
            weighted_kernel=config.weighted_kernel,
            subsample_train=config.subsample_train,
            subsample_eval=config.subsample_eval,
        )
    elif config.model == "GNOFNOGNOTrackB":
        print("using GNOFNOGNOTrackB")
        model = GNOFNOGNOTrackB(
            radius_in=config.radius_in,
            radius_out=config.radius_out,
            embed_dim=32,
            hidden_channels=(64, 64),
            in_channels=1,
            out_channels=1,
            fno_modes=(32, 32, 32),
            fno_hidden_channels=64,
            fno_out_channels=64,
            fno_domain_padding=0.125,
            fno_norm="group_norm",
            fno_factorization="tucker",
            fno_rank=0.4,
            linear_kernel=True,
            weighted_kernel=config.weighted_kernel,
            subsample_train=config.subsample_train,
            subsample_eval=config.subsample_eval,
        )
    elif config.model == "FNOGNO":
        print("using FNOGNO")
        model = FNOGNO(radius=config.radius)
    elif config.model == "FNOGNOAhmed":
        print("using FNOGNOAhmed")
        model = FNOGNO(radius=config.radius)
    elif config.model == "Transolver":
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
    elif config.model == "Transolver_conv_sdf":
        print("using Transolver_conv_sdf")
        model = Transolver_conv_sdf(
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
    elif config.model == "Transolver_conv_multifi":
        print("using Transolver_conv_multifi")
        model = Transolver_conv_multifi(
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
    elif config.model == "TransGINO":
        print("using TransGINO")
        model = TransGINO(
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
            n_modes=config.fno_modes,
            hidden_channels=config.hidden_channels,
            norm=config.norm,
            use_mlp=config.use_mlp,
            mlp={"expansion": 1.0, "dropout": 0},
            domain_padding=config.domain_padding,
            factorization=config.factorization,
            rank=0.4,
            out_channels=out_channels,
            r=config.r,
            resolution=config.sdf_spatial_resolution,
            gno_implementation="torch_scatter",
        )
    else:
        raise ValueError("Network not supported")

    # print(model)  # 这一句显示网络结构
    print("The model size is ", count_params(model))
    return model
