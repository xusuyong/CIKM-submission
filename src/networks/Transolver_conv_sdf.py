import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
from .base_model import BaseModel
from neuralop.models.tfno import Lifting, Projection
# import torch
# import torch.nn as nn
# from .base_model import BaseModel
from .neighbor_ops import (
    NeighborSearchLayer,
    NeighborMLPConvLayer,
    NeighborMLPConvLayerLinear,
    NeighborMLPConvLayerWeighted,
)

# from neuralop.models import FNO
# from neuralop.models.tfno import Projection

# from .net_utils import PositionalEmbedding, AdaIN, MLP

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
import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_
# from model.Embedding import timestep_embedding
import torch.utils.checkpoint as checkpoint
# from model.Physics_Attention import Physics_Attention_Structured_Mesh_3D


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

class Physics_Attention_Structured_Mesh_3D(nn.Module):
    ## for structured mesh in 3D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=32, H=32, W=32, D=32, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.D = D

        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, self.D, C).contiguous().permute(0, 4, 1, 2, 3).contiguous()  # B C H W

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
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
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block_3D(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            H=32,
            W=32,
            D=32
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Structured_Mesh_3D(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                         dropout=dropout, slice_num=slice_num, H=H, W=W, D=D)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
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


class Model_3D(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 n_head=8,
                 Time_Input=False,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False,
                 H=32,
                 W=32,
                 D=32,
                 ):
        super(Model_3D, self).__init__()
        self.__name__ = 'Transolver_3D'
        self.use_checkpoint = False
        self.H = H
        self.W = W
        self.D = D
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.pos = self.get_grid()
            self.preprocess = MLP(fun_dim + self.ref * self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0,
                                  res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.ModuleList([Transolver_block_3D(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      H=H,
                                                      W=W,
                                                      D=D,
                                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

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

    def get_grid(self, batchsize=1):
        size_x, size_y, size_z = self.H, self.W, self.D
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        grid = torch.cat((gridx, gridy, gridz), dim=-1).cuda()  # B H W D 3

        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).cuda()  # B 4 4 4 3

        pos = torch.sqrt(
            torch.sum((grid[:, :, :, :, None, None, None, :] - grid_ref[:, None, None, None, :, :, :, :]) ** 2,
                      dim=-1)). \
            reshape(batchsize, size_x, size_y, size_z, self.ref * self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, fx, T=None):
        if self.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1, 1, 1).reshape(x.shape[0], self.H * self.W * self.D,
                                                                self.ref * self.ref * self.ref)
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        # if T is not None:
        #     Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
        #     Time_emb = self.time_fc(Time_emb)
        #     fx = fx + Time_emb

        for block in self.blocks:
            if self.use_checkpoint:
                fx = checkpoint.checkpoint(block, fx)
            else:
                fx = block(fx)

        return fx


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
            self.projection = Projection(
                in_channels=hidden_dim,
                out_channels=out_dim,
                hidden_channels=256,
                non_linearity=nn.functional.gelu,
                n_dim=1,
            )

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            # return self.mlp2(self.ln_3(fx))
            aaa = self.ln_3(fx)
            bbb = self.projection(aaa.permute(0, 2, 1)).permute(0, 2, 1)
            return bbb
        else:
            return fx


class Transolver_conv_sdf(BaseModel):
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
        super(Transolver_conv_sdf, self).__init__()
        self.__name__ = "UniPDE_3D"
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.preprocess = Lifting(
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
            self.lifting = Lifting(
                in_channels=fun_dim + space_dim, out_channels=n_hidden, n_dim=1
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

    def forward(self, data, sdf_query_points, sdf):
        decode_sdf = Model_3D(sdf_query_points, sdf)

        neighbors = self.nsearch(grid.squeeze(), vert.squeeze())  # len 3
        vert_sdf = self.graph_conv(
            in_features=grid_sdf, out_features=vert_sdf, neighbors=neighbors
        )  # torch.Size([3586, 32])
        vert_sdf = vert_sdf.permute(1, 0).unsqueeze(0)  # torch.Size([1, 32, 3586])

        # u = u.unsqueeze(0).permute(0, 2, 1)  # (1, hidden_channels[1], n_in/n_eval)  torch.Size([1, 64, 3586])

        # Pointwise projection to out channels
        # u = self.projection(u).squeeze(0).permute(1, 0)  # (n_in/n_eval, out_channels)  torch.Size([3586, 1])



        x, fx, T = data, None, None  # torch.Size([1, 3586, 6])
        # x = x[-3682:, :] #torch.Size([3682, 7])
        # x = torch.cat((x[0:16], x[112:]), dim=0) # torch.Size([3586, 7]) 因为被去掉的点不是表面点
        # x = torch.cat((x[:, :3], x[:, 4:]), dim=1) #torch.Size([3586, 6]) 因为表面的SDF全是0所以去掉，但这里不是0因为归一化过了？
        # x = x[None, :, :] #torch.Size([1, 3586, 6])

        if self.unified_pos:  # 不执行
            new_pos = self.get_grid(data.pos[None, :, :])
            x = torch.cat((x, new_pos), dim=-1)
        if x.shape[0] != 1:
            x = x.unsqueeze(0)
        if fx is not None:  # 不执行
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:  # 执行
            # fx = self.preprocess(x)  # torch.Size([1, 3586, 256])
            fx = self.lifting(x.permute(0, 2, 1)).permute(
                0, 2, 1
            )  # torch.Size([1, 3586, 256])
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
            sdf_query_points = data_dict["df_query_points"].squeeze(0).permute(1, 2, 3, 0).to(self.device)
            sdf = data_dict["df"].to(self.device)
            return ca, sdf_query_points, sdf
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
        vert_normal, sdf_query_points, sdf = self.data_dict_to_input(data_dict)
        vert_normal = vert_normal[:: self.subsample_train]
        pressure = self(vert_normal, sdf_query_points, sdf)
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

    # def loss_dict(self, data_dict, loss_fn=None):
    #     vert_normal = self.data_dict_to_input(data_dict)
    #     vert_normal_s1 = vert_normal[:: self.subsample_train]
    #     vert_normal_s2 = vert_normal[1 :: self.subsample_train]
    #     pressure_s1 = self(vert_normal_s1)
    #     pressure_s2 = self(vert_normal_s2)
    #     pressure = torch.cat([pressure_s1, pressure_s2], dim=1)
    #     if loss_fn is None:
    #         loss_fn = self.loss
    #     # loss_fn = torch.nn.MSELoss(reduction="mean")

    #     if isinstance(data_dict["pressure"], list):
    #         truth_s1 = data_dict["pressure"][0][:: self.subsample_train]
    #         truth_s2 = data_dict["pressure"][0][1 :: self.subsample_train]
    #         truth = torch.cat([truth_s1, truth_s2], dim=0)
    #         return {"loss": loss_fn(pressure.squeeze(-1), truth.to(self.device))}
    #     else:
    #         return {
    #             "loss": loss_fn(
    #                 pressure.squeeze(-1), data_dict["pressure"].to(self.device)
    #             )
    #         }
