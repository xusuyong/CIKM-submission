import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
from .base_model import BaseModel

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
            hidden_dim,#输入
            hidden_dim * mlp_ratio,#隐藏
            hidden_dim,#输出
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
        
        self.finalmlp = MLP(
                        fun_dim + space_dim,
                        n_hidden * 2,
                        n_hidden,
                        n_layers=0,
                        res=False,
                        act=act,
                    )
        # Fully connected layers to interpret the extracted features and make predictions
        args={}
        args['dropout']=0.4
        args['emb_dims']=256
        self.linear1 = nn.Linear(args['emb_dims']*2, 128, bias=False)
        self.bn6 = nn.BatchNorm1d(128)
        self.dp1 = nn.Dropout(p=args['dropout'])

        self.linear2 = nn.Linear(128, 64)
        self.bn7 = nn.BatchNorm1d(64)
        self.dp2 = nn.Dropout(p=args['dropout'])

        self.linear3 = nn.Linear(64, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.dp3 = nn.Dropout(p=args['dropout'])

        self.linear4 = nn.Linear(32, 16)
        self.bn9 = nn.BatchNorm1d(16)
        self.dp4 = nn.Dropout(p=args['dropout'])

        self.linear5 = nn.Linear(16, 1)  # The final output layer
        
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
            new_pos = self.get_grid(data.pos[None, :, :])
            x = torch.cat((x, new_pos), dim=-1)

        if fx is not None:  # 不执行
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:  # 执行
            fx = self.preprocess(x)  # torch.Size([1, 3586, 256])
            fx = fx + self.placeholder[None, None, :]  # torch.Size([1, 3586, 256])

        for block in self.blocks:
            fx = block(fx)
            
        # x1 = F.adaptive_max_pool1d(fx.permute(0,2,1), 1).view(1, -1)
        # # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        # x2 = F.adaptive_avg_pool1d(fx, 1).view(1, -1)
        
        # # Process features through fully connected layers with dropout and batch normalization
        # fx = F.leaky_relu(self.bn6(self.linear1(fx)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 128)
        # fx = self.dp1(fx)
        # fx = F.leaky_relu(self.bn7(self.linear2(fx)), negative_slope=0.2)  # (batch_size, 128) -> (batch_size, 64)
        # fx = self.dp2(fx)
        # fx = F.leaky_relu(self.bn8(self.linear3(fx)), negative_slope=0.2)  # (batch_size, 64) -> (batch_size, 32)
        # fx = self.dp3(fx)
        # fx = F.leaky_relu(self.bn9(self.linear4(fx)), negative_slope=0.2)  # (batch_size, 32) -> (batch_size, 16)
        # fx = self.dp4(fx)

        # # Final linear layer to produce the output
        # fx = self.linear5(fx)                                              # (batch_size, 16) -> (batch_size, 1)

        return fx  # 返回第一个样本（因为batchsize是1？？） torch.Size([1, 3586, 1])

    def data_dict_to_input(self, data_dict, **kwargs):
        if "vert_normals" in data_dict.keys():
            vert_normals = data_dict["vert_normals"].to(self.device)
            return vert_normals
        elif "cd" in data_dict.keys():
            centroids = data_dict["centroids"][0].to(self.device)
            areas = data_dict["areas"][0].unsqueeze(-1).to(self.device)
            normal = data_dict["normal"][0].to(self.device)
            ca = torch.cat([centroids, areas, normal], dim=1)
            return ca
        elif "centroids" in data_dict.keys() and "areas" in data_dict.keys():
            centroids = data_dict["centroids"][0].to(self.device)
            areas = data_dict["areas"][0].unsqueeze(-1).to(self.device)
            ca = torch.cat([centroids, areas], dim=1)
            return ca
        elif "vertices" in data_dict.keys():
            vert = data_dict["vertices"].to(self.device)
            return vert

    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None):
        vert = self.data_dict_to_input(data_dict)
        pred_var = self(vert)
        if "pressure" in data_dict.keys():
            if isinstance(data_dict["pressure"], list):
                gt_out = data_dict["pressure"][0]
            else:
                gt_out = data_dict["pressure"]
            pred_var_key = "pressure"
        elif "velocity" in data_dict.keys():
            gt_out = data_dict["velocity"]
            pred_var_key = "velocity"
        elif "cd" in data_dict.keys():
            pred_var = torch.mean(pred_var).reshape(1,1)
            gt_out = data_dict["cd"]
            pred_var_key = "cd"
        else:
            raise NotImplementedError("only pressure velocity works")

        out_loss_dict = {"l2": loss_fn(pred_var, gt_out.to(self.device))}
        if decode_fn is not None:
            pred_var = decode_fn(pred_var)
            gt_out = decode_fn(gt_out)
            out_loss_dict["l2_decoded"] = loss_fn(pred_var, gt_out)
        output_dict = {pred_var_key: pred_var}
        return out_loss_dict, output_dict

    def loss_dict(self, data_dict, loss_fn=None):
        vert_normal = self.data_dict_to_input(data_dict)
        vert_normal = vert_normal[:: self.subsample_train]
        pred_var = self(vert_normal)
        if loss_fn is None:
            loss_fn = self.loss
        # loss_fn = torch.nn.MSELoss(reduction="mean")
        if "pressure" in data_dict.keys():
            if isinstance(data_dict["pressure"], list):
                true_var = data_dict["pressure"][0][:: self.subsample_train]
            else:
                true_var = data_dict["pressure"]
        elif "velocity" in data_dict.keys():
            true_var = data_dict["velocity"]
        elif "cd" in data_dict.keys():
            pred_var = torch.mean(pred_var).reshape(1,1)
            true_var = data_dict["cd"]
        else:
            raise NotImplementedError("only pressure velocity works")

        return {"loss": loss_fn(pred_var.squeeze(-1), true_var.to(self.device))}

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


class TransolvertrackB(Transolver):
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
        if "vertices" in data_dict:
            vert = data_dict["vertices"].to(self.device)
            return vert
        elif "centroids" in data_dict and "areas" in data_dict:
            centroids = data_dict["centroids"][0].to(self.device)
            areas = data_dict["areas"][0].unsqueeze(-1).to(self.device)
            ca = torch.cat([centroids, areas], dim=1)
            return ca
        elif "vert_normals" in data_dict:
            vert_normals = data_dict["vert_normals"].to(self.device)
            return vert_normals

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

    # def loss_dict(self, data_dict, loss_fn=None):
    #     vert_normal = self.data_dict_to_input(data_dict)
    #     vert_normal = vert_normal[:: self.subsample_train]
    #     pressure = self(vert_normal)
    #     if loss_fn is None:
    #         loss_fn = self.loss
    #     # loss_fn = torch.nn.MSELoss(reduction="mean")

    #     if isinstance(data_dict["pressure"], list):
    #         truth = data_dict["pressure"][0][:: self.subsample_train]
    #         return {"loss": loss_fn(pressure.squeeze(-1), truth.to(self.device))}
    #     else:
    #         return {
    #             "loss": loss_fn(
    #                 pressure.squeeze(-1), data_dict["pressure"].to(self.device)
    #             )
    #         }

    def loss_dict(self, data_dict, loss_fn=None):
        vert_normal = self.data_dict_to_input(data_dict)
        vert_normal_s1 = vert_normal[:: self.subsample_train]
        vert_normal_s2 = vert_normal[1 :: self.subsample_train]
        pressure_s1 = self(vert_normal_s1)
        pressure_s2 = self(vert_normal_s2)
        pressure = torch.cat([pressure_s1, pressure_s2], dim=0)
        if loss_fn is None:
            loss_fn = self.loss
        # loss_fn = torch.nn.MSELoss(reduction="mean")

        if isinstance(data_dict["pressure"], list):
            truth_s1 = data_dict["pressure"][0][:: self.subsample_train]
            truth_s2 = data_dict["pressure"][0][1 :: self.subsample_train]
            truth = torch.cat([truth_s1, truth_s2], dim=0)
            return {"loss": loss_fn(pressure.squeeze(-1), truth.to(self.device))}
        else:
            return {
                "loss": loss_fn(
                    pressure.squeeze(-1), data_dict["pressure"].to(self.device)
                )
            }


class TransolverVelocity(Transolver):
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
            new_pos = self.get_grid(data.pos[None, :, :])
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


class Transolver_Cd(BaseModel):
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
        super(Transolver_Cd, self).__init__()
        self.__name__ = "UniPDE_3D"
        self.ref = ref
        self.unified_pos = unified_pos
        self.num_points = 50000
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
                    out_dim=256,
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
        
        self.finalmlp = MLP(
                        fun_dim + space_dim,
                        n_hidden * 2,
                        n_hidden,
                        n_layers=0,
                        res=False,
                        act=act,
                    )
        # Fully connected layers to interpret the extracted features and make predictions
        args={}
        args['dropout']=0.4
        args['emb_dims']=256
        self.linear1 = nn.Linear(args['emb_dims']*2, 128, bias=False)
        self.bn6 = nn.BatchNorm1d(128)
        self.dp1 = nn.Dropout(p=args['dropout'])

        self.linear2 = nn.Linear(128, 64)
        self.bn7 = nn.BatchNorm1d(64)
        self.dp2 = nn.Dropout(p=args['dropout'])

        self.linear3 = nn.Linear(64, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.dp3 = nn.Dropout(p=args['dropout'])

        self.linear4 = nn.Linear(32, 16)
        self.bn9 = nn.BatchNorm1d(16)
        self.dp4 = nn.Dropout(p=args['dropout'])

        self.linear5 = nn.Linear(16, 1)  # The final output layer
        
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
        x, fx, T = data, None, None


        if self.unified_pos:  # 不执行
            new_pos = self.get_grid(data.pos[None, :, :])
            x = torch.cat((x, new_pos), dim=-1)

        if fx is not None:  # 不执行
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:  # 执行
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)
        fx=fx.permute(0,2,1)
        x1 = F.adaptive_max_pool1d(fx, 1).view(1, -1)
        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(fx, 1).view(1, -1)
        fx = torch.cat((x1, x2), 1)   # (batch_size, emb_dims*2)
        
        """BN层的BS要大于1"""
        # # Process features through fully connected layers with dropout and batch normalization
        # fx = F.leaky_relu(self.bn6(self.linear1(fx)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 128)
        # fx = self.dp1(fx)
        # fx = F.leaky_relu(self.bn7(self.linear2(fx)), negative_slope=0.2)  # (batch_size, 128) -> (batch_size, 64)
        # fx = self.dp2(fx)
        # fx = F.leaky_relu(self.bn8(self.linear3(fx)), negative_slope=0.2)  # (batch_size, 64) -> (batch_size, 32)
        # fx = self.dp3(fx)
        # fx = F.leaky_relu(self.bn9(self.linear4(fx)), negative_slope=0.2)  # (batch_size, 32) -> (batch_size, 16)
        # fx = self.dp4(fx)


        # Process features through fully connected layers with dropout and batch normalization
        fx = F.leaky_relu(self.linear1(fx), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 128)
        fx = self.dp1(fx)
        fx = F.leaky_relu(self.linear2(fx), negative_slope=0.2)  # (batch_size, 128) -> (batch_size, 64)
        fx = self.dp2(fx)
        fx = F.leaky_relu(self.linear3(fx), negative_slope=0.2)  # (batch_size, 64) -> (batch_size, 32)
        fx = self.dp3(fx)
        fx = F.leaky_relu(self.linear4(fx), negative_slope=0.2)  # (batch_size, 32) -> (batch_size, 16)
        fx = self.dp4(fx)
        
        # Final linear layer to produce the output
        fx = self.linear5(fx)                                              # (batch_size, 16) -> (batch_size, 1)

        return fx  # 返回第一个样本（因为batchsize是1？？） torch.Size([1, 3586, 1])

    def data_dict_to_input(self, data_dict, **kwargs):
        if "vert_normals" in data_dict.keys():
            vert_normals = data_dict["vert_normals"].to(self.device)
            return vert_normals
        elif "cd" in data_dict.keys():
            centroids = data_dict["centroids"][0].to(self.device)
            areas = data_dict["areas"][0].unsqueeze(-1).to(self.device)
            normal = data_dict["normal"][0].to(self.device)
            ca = torch.cat([centroids, areas, normal], dim=1)
            return ca
        elif "centroids" in data_dict.keys() and "areas" in data_dict.keys():
            centroids = data_dict["centroids"][0].to(self.device)
            areas = data_dict["areas"][0].unsqueeze(-1).to(self.device)
            ca = torch.cat([centroids, areas], dim=1)
            return ca
        elif "vertices" in data_dict.keys():
            vert = data_dict["vertices"].to(self.device)
            return vert

    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None):
        vert = self.data_dict_to_input(data_dict)
        pred_var = self(vert)
        if "pressure" in data_dict.keys():
            if isinstance(data_dict["pressure"], list):
                gt_out = data_dict["pressure"][0]
            else:
                gt_out = data_dict["pressure"]
            pred_var_key = "pressure"
        elif "velocity" in data_dict.keys():
            gt_out = data_dict["velocity"]
            pred_var_key = "velocity"
        elif "cd" in data_dict.keys():
            pred_var = torch.mean(pred_var).reshape(1,1)
            gt_out = data_dict["cd"]
            pred_var_key = "cd"
        else:
            raise NotImplementedError("only pressure velocity works")

        out_loss_dict = {"l2": loss_fn(pred_var, gt_out.to(self.device))}
        if decode_fn is not None:
            pred_var = decode_fn(pred_var)
            gt_out = decode_fn(gt_out)
            out_loss_dict["l2_decoded"] = loss_fn(pred_var, gt_out)
        output_dict = {pred_var_key: pred_var}
        return out_loss_dict, output_dict

    def _sample_or_pad_vertices(self, vertices: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        Subsamples or pads the vertices of the model to a fixed number of points.

        Args:
            vertices: The vertices of the 3D model as a torch.Tensor.
            num_points: The desired number of points for the model.

        Returns:
            The vertices standardized to the specified number of points.
        """
        num_vertices = vertices.size(0)
        # Subsample the vertices if there are more than the desired number
        if num_vertices > num_points:
            indices = np.random.choice(num_vertices, num_points, replace=False)
            vertices = vertices[indices]
        # Pad with zeros if there are fewer vertices than desired
        elif num_vertices < num_points:
            padding = torch.zeros((num_points - num_vertices, 3), dtype=torch.float32).to(self.device)
            vertices = torch.cat((vertices, padding), dim=0)#仅仅是pad成0了
        return vertices

    def loss_dict(self, data_dict, loss_fn=None):
        vert_normal = self.data_dict_to_input(data_dict)
        # vert_normal = self._sample_or_pad_vertices(vert_normal, self.num_points)#torch.Size([500000, 3])
        # vert_normal = vert_normal[:: self.subsample_train]
        pred_var = self(vert_normal)
        if loss_fn is None:
            loss_fn = self.loss
        # loss_fn = torch.nn.MSELoss(reduction="mean")
        if "pressure" in data_dict.keys():
            if isinstance(data_dict["pressure"], list):
                true_var = data_dict["pressure"][0][:: self.subsample_train]
            else:
                true_var = data_dict["pressure"]
        elif "velocity" in data_dict.keys():
            true_var = data_dict["velocity"]
        elif "cd" in data_dict.keys():
            true_var = data_dict["cd"]
        else:
            raise NotImplementedError("only pressure velocity works")

        return {"loss": loss_fn(pred_var.squeeze(-1), true_var.to(self.device))}