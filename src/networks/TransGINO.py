import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
from .base_model import BaseModel


import torch
import torch.nn as nn

from neuralop.models import FNO
from neuralop.models.tfno import Projection
from neuralop.models.spectral_convolution import FactorizedSpectralConv

from torch_geometric.nn import NNConv
from .net_utils import MLP

from .neighbor_ops import NeighborSearchLayer, NeighborMLPConvLayer


ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class Physics_Attention_1D(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        # B N C
        B, N, C = x.shape

        ### (1) Slice
        fx_mid = (
            self.in_project_fx(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N C   torch.Size([1, 8, 32186, 32])
        x_mid = (
            self.in_project_x(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N C   torch.Size([1, 8, 32186, 32])
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / self.temperature
        )  # B H N G torch.Size([1, 8, 32186, 32])
        slice_norm = slice_weights.sum(2)  # B H G  torch.Size([1, 8, 32])
        slice_token = torch.einsum(
            "bhnc,bhng->bhgc", fx_mid, slice_weights
        )  # torch.Size([1, 8, 32, 32])
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )  # torch.Size([1, 8, 32, 32])

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)  # torch.Size([1, 8, 32, 32])
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(
            attn, v_slice_token
        )  # B H G D  torch.Size([1, 8, 32, 32])

        ### (3) Deslice
        out_x = torch.einsum(
            "bhgc,bhng->bhnc", out_slice_token, slice_weights
        )  # torch.Size([1, 8, 32186, 32])
        out_x = rearrange(out_x, "b h n d -> b n (h d)")  # torch.Size([1, 32186, 256])
        return self.to_out(out_x)  # torch.Size([1, 32186, 256])


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(n_hidden, n_hidden), act())
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        # print(x)
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_1D(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Transolver(BaseModel):
    def __init__(
        self,
        space_dim=1,
        n_layers=5,
        n_hidden=256,
        dropout=0,
        n_head=8,
        act="gelu",
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        slice_num=32,
        ref=8,
        unified_pos=False,
        subsample_train=1,
        subsample_eval=1,
    ):
        self.subsample_train = subsample_train
        self.subsample_eval = subsample_eval
        super(Transolver, self).__init__()
        self.__name__ = "UniPDE_3D"
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.preprocess = MLP(
                fun_dim + self.ref * self.ref * self.ref,
                n_hidden * 2,
                n_hidden,
                n_layers=0,
                res=False,
                act=act,
            )
        else:
            self.preprocess = MLP(
                fun_dim + space_dim,
                n_hidden * 2,
                n_hidden,
                n_layers=0,
                res=False,
                act=act,
            )

        self.n_hidden = n_hidden
        self.space_dim = space_dim

        self.blocks = nn.ModuleList(
            [
                Transolver_block(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    last_layer=(_ == n_layers - 1),
                )
                for _ in range(n_layers)
            ]
        )
        self.initialize_weights()
        self.placeholder = nn.Parameter(
            (1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float)
        )

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, my_pos):
        # my_pos 1 N 3
        batchsize = my_pos.shape[0]

        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat(
            [batchsize, 1, self.ref, self.ref, 1]
        )
        gridy = torch.tensor(np.linspace(0, 2, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat(
            [batchsize, self.ref, 1, self.ref, 1]
        )
        gridz = torch.tensor(np.linspace(-4, 4, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat(
            [batchsize, self.ref, self.ref, 1, 1]
        )
        grid_ref = (
            torch.cat((gridx, gridy, gridz), dim=-1)
            .cuda()
            .reshape(batchsize, self.ref**3, 3)
        )  # B 4 4 4 3

        pos = (
            torch.sqrt(
                torch.sum(
                    (my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1
                )
            )
            .reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref)
            .contiguous()
        )
        return pos

    def forward(self, data):
        # cfd_data = data
        x, fx, T = data, None, None  # torch.Size([1, 3586, 6])
        # x = x[-3682:, :] #torch.Size([3682, 7])
        # x = torch.cat((x[0:16], x[112:]), dim=0) # torch.Size([3586, 7]) 因为被去掉的点不是表面点
        # x = torch.cat((x[:, :3], x[:, 4:]), dim=1) #torch.Size([3586, 6]) 因为表面的SDF全是0所以去掉，但这里不是0因为归一化过了？
        # x = x[None, :, :] #torch.Size([1, 3586, 6])

        if self.unified_pos:  # 不执行
            new_pos = self.get_grid(cfd_data.pos[None, :, :])
            x = torch.cat((x, new_pos), dim=-1)

        if fx is not None:  # 不执行
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:  # 执行
            fx = self.preprocess(x)  # torch.Size([1, 3586, 256])
            fx = fx + self.placeholder[None, None, :]  # torch.Size([1, 3586, 256])

        for block in self.blocks:
            fx = block(fx)

        return fx  # 返回第一个样本（因为batchsize是1？？） torch.Size([1, 3586, 1])

    def data_dict_to_input(self, data_dict, **kwargs):
        if "vert_normals" in data_dict:
            vert_normals = data_dict["vert_normals"].to(self.device)
            return vert_normals
        elif "centroids" in data_dict and "areas" in data_dict:
            centroids = data_dict["centroids"][0].to(self.device)
            areas = data_dict["areas"][0].unsqueeze(-1).to(self.device)
            ca = torch.cat([centroids, areas], dim=1)
            return ca
        elif "vertices" in data_dict:
            vert = data_dict["vertices"].to(self.device)
            return vert

    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None):
        vert = self.data_dict_to_input(data_dict)
        pred_out = self(vert)
        if isinstance(data_dict["pressure"], list):
            gt_out = data_dict["pressure"][0].to(self.device)
        else:
            gt_out = data_dict["pressure"].to(self.device)
        out_dict = {"l2": loss_fn(pred_out, gt_out)}
        if decode_fn is not None:
            pred_out = decode_fn(pred_out)
            gt_out = decode_fn(gt_out)
            out_dict["l2_decoded"] = loss_fn(pred_out, gt_out)
        return out_dict

    def loss_dict(self, data_dict, loss_fn=None):
        vert_normal = self.data_dict_to_input(data_dict)
        vert_normal = vert_normal[:: self.subsample_train]
        pressure = self(vert_normal)
        if loss_fn is None:
            loss_fn = self.loss
        # loss_fn = torch.nn.MSELoss(reduction="mean")

        if isinstance(data_dict["pressure"], list):
            truth = data_dict["pressure"][0][:: self.subsample_train]
            return {"loss": loss_fn(pressure.squeeze(-1), truth.to(self.device))}
        else:
            return {
                "loss": loss_fn(
                    pressure.squeeze(-1), data_dict["pressure"].to(self.device)
                )
            }


class TransGINO(BaseModel):
    def __init__(
        self,
        space_dim=1,
        n_layers=5,
        n_hidden=256,
        dropout=0,
        n_head=8,
        act="gelu",
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        slice_num=32,
        ref=8,
        unified_pos=False,
        subsample_train=1,
        subsample_eval=1,
        n_modes=[24, 24, 24],
        hidden_channels=32,
        in_channels=4,
        out_channels=1,
        lifting_channels=64,
        projection_channels=64,
        n_layers_fno=4,
        interp_mode="interp_before",
        incremental_n_modes=None,
        use_mlp=False,
        mlp=None,
        non_linearity=torch.nn.functional.gelu,
        norm=None,
        preactivation=False,
        fno_skip="linear",
        mlp_skip="soft-gating",
        separable=False,
        factorization=None,
        rank=1,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        fft_norm="forward",
        SpectralConv=FactorizedSpectralConv,
        resolution=[64, 64, 64],
        r=0.05,
        gno_implementation="torch_scatter",
        **kwargs
    ):
        super().__init__()
        self.subsample_train = subsample_train
        self.subsample_eval = subsample_eval

        self.fno = FNO(
            n_modes,  # len()3个24
            hidden_channels,  # 32
            in_channels,  # 4
            out_channels,  # 1
            lifting_channels,  # 64
            projection_channels,  # 64
            n_layers_fno,  # 4
            None,
            incremental_n_modes,  # None
            use_mlp,  # true
            0,
            0.5,
            # mlp,
            non_linearity,
            norm,  # "group_norm"
            preactivation,  # false
            fno_skip,  # linear
            mlp_skip,  # "soft-gating"
            separable,  # false
            factorization,  # tucker
            rank,  # 0.4
            joint_factorization,  # false
            fixed_rank_modes,  # false
            implementation,  # factorize
            decomposition_kwargs,
            domain_padding,  # 0.125
            domain_padding_mode,  # one side
            fft_norm,  # forward
            SpectralConv,
            **kwargs
        )

        self.projection = Projection(
            in_channels=64**3,
            out_channels=out_channels,
            hidden_channels=256,
            non_linearity=nn.functional.gelu,
            n_dim=1,
        )

        self.__name__ = "UniPDE_3D"
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.preprocess = MLP(
                fun_dim + self.ref * self.ref * self.ref,
                n_hidden * 2,
                n_hidden,
                n_layers=0,
                res=False,
                act=act,
            )
        else:
            self.preprocess = MLP(
                fun_dim + space_dim,
                n_hidden * 2,
                n_hidden,
                n_layers=0,
                res=False,
                act=act,
            )

        self.n_hidden = n_hidden
        self.space_dim = space_dim

        self.blocks = nn.ModuleList(
            [
                Transolver_block(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    # last_layer=False,
                    last_layer=(_ == n_layers - 1),
                )
                for _ in range(n_layers)
            ]
        )
        self.initialize_weights()
        self.placeholder = nn.Parameter(
            (1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float)
        )

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, my_pos):
        # my_pos 1 N 3
        batchsize = my_pos.shape[0]

        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat(
            [batchsize, 1, self.ref, self.ref, 1]
        )
        gridy = torch.tensor(np.linspace(0, 2, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat(
            [batchsize, self.ref, 1, self.ref, 1]
        )
        gridz = torch.tensor(np.linspace(-4, 4, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat(
            [batchsize, self.ref, self.ref, 1, 1]
        )
        grid_ref = (
            torch.cat((gridx, gridy, gridz), dim=-1)
            .cuda()
            .reshape(batchsize, self.ref**3, 3)
        )  # B 4 4 4 3

        pos = (
            torch.sqrt(
                torch.sum(
                    (my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1
                )
            )
            .reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref)
            .contiguous()
        )
        return pos

    def forward(self, vert, sdf, sdf_query):
        x, fx, T = vert, None, None  # torch.Size([1, 3586, 6])

        # if self.unified_pos:  # 不执行
        # new_pos = self.get_grid(cfd_data.pos[None, :, :])
        # x = torch.cat((x, new_pos), dim=-1)

        if fx is not None:  # 不执行
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:  # 执行
            fx = self.preprocess(x)  # torch.Size([1, 3586, 256])
            fx = fx + self.placeholder[None, None, :]  # torch.Size([1, 3586, 256])

        for block in self.blocks:
            fx = block(fx)  # TODO 最后一层不要MLP

        resolution = sdf.shape[-1]  # 16,32,64

        # sdf_query_points和sdf的组合([1, 4, 32, 32, 32]) 顶点([1, 3586, 3])
        """TFNO's forward pass"""

        grid_sdf = torch.cat((sdf_query, sdf.unsqueeze(1)), dim=1).to(
            self.device
        )  # torch.Size([1, 4, 32, 32, 32]) 把sdf_query_points和sdf组合起来了

        grid_sdf = self.fno.lifting(grid_sdf)  # torch.Size([1, 32, 32, 32, 32])

        if self.fno.domain_padding is not None:
            grid_sdf = self.fno.domain_padding.pad(
                grid_sdf
            )  # torch.Size([1, 32, 36, 36, 36])

        for layer_idx in range(self.fno.n_layers):
            grid_sdf = self.fno.fno_blocks(
                grid_sdf, layer_idx
            )  # torch.Size([1, 32, 36, 36, 36])

        if self.fno.domain_padding is not None:
            grid_sdf = self.fno.domain_padding.unpad(
                grid_sdf
            )  # torch.Size([1, 32, 64, 64, 64])

        # grid_sdf = self.fno(grid_sdf)  # TODO 检查padding对不对
        grid_sdf = (
            grid_sdf.squeeze().permute(1, 2, 3, 0).reshape(resolution**3, -1)
        )  # (n_x*n_y*n_z, fno_out_channels)  torch.Size([262144, 64])
        fx = fx.squeeze()
        y = torch.einsum("bi,ni->bn", fx, grid_sdf)
        # y = torch.einsum("bi,ni->bi", fx, grid_sdf)

        y = y.unsqueeze(0).permute(
            0, 2, 1
        )  # (1, hidden_channels[1], n_in/n_eval)  torch.Size([1, 64, 3586])

        # Pointwise projection to out channels
        y = (
            self.projection(y).squeeze(0).permute(1, 0)
        )  # (N, out_channels)  torch.Size([3586, 1])

        return y

    def data_dict_to_input(self, data_dict, **kwargs):
        if "vert_normals" in data_dict:
            vert_normals = data_dict["vert_normals"].to(self.device)
            return vert_normals
        elif "centroids" in data_dict and "areas" in data_dict:
            centroids = data_dict["centroids"][0].to(self.device)
            areas = data_dict["areas"][0].unsqueeze(-1).to(self.device)
            ca = torch.cat([centroids, areas], dim=1)
            return ca
        elif "vertices" in data_dict:
            vert = data_dict["vertices"].to(self.device)
            sdf = data_dict["sdf"].to(self.device)
            sdf_query = data_dict["sdf_query_points"].to(self.device)
            return vert, sdf, sdf_query

    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None):
        vert, sdf, sdf_query = self.data_dict_to_input(data_dict)
        pred_out = self(vert, sdf, sdf_query)
        if isinstance(data_dict["pressure"], list):
            gt_out = data_dict["pressure"][0].to(self.device)
        else:
            gt_out = data_dict["pressure"].to(self.device)
        out_dict = {"l2": loss_fn(pred_out, gt_out)}
        if decode_fn is not None:
            pred_out = decode_fn(pred_out)
            gt_out = decode_fn(gt_out)
            out_dict["l2_decoded"] = loss_fn(pred_out, gt_out)
        return out_dict

    def loss_dict(self, data_dict, loss_fn=None):
        vert, sdf, sdf_query = self.data_dict_to_input(data_dict)
        vert = vert[:: self.subsample_train]
        sdf = sdf[:: self.subsample_train]
        sdf_query = sdf_query[:: self.subsample_train]
        pressure = self(vert, sdf, sdf_query)
        if loss_fn is None:
            loss_fn = self.loss
        # loss_fn = torch.nn.MSELoss(reduction="mean")

        if isinstance(data_dict["pressure"], list):
            truth = data_dict["pressure"][0][:: self.subsample_train]
            return {"loss": loss_fn(pressure.squeeze(-1), truth.to(self.device))}
        else:
            return {
                "loss": loss_fn(
                    pressure.squeeze(-1), data_dict["pressure"].to(self.device)
                )
            }
