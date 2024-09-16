import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import copy
import math
import numpy as np
from .base_model import BaseModel

def knn(x, k):
    """
    Computes the k-nearest neighbors for each point in x.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, num_dims, num_points).
        k (int): The number of nearest neighbors to find.

    Returns:
        torch.Tensor: Indices of the k-nearest neighbors for each point, shape (batch_size, num_points, k).
    """
    # Calculate pairwise distance, shape (batch_size, num_points, num_points)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # Retrieve the indices of the k nearest neighbors
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Constructs local graph features for each point by finding its k-nearest neighbors and
    concatenating the relative position vectors.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, num_dims, num_points).
        k (int): The number of neighbors to consider for graph construction.
        idx (torch.Tensor, optional): Precomputed k-nearest neighbor indices.

    Returns:
        torch.Tensor: The constructed graph features of shape (batch_size, 2*num_dims, num_points, k).
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    # Compute k-nearest neighbors if not provided
    if idx is None:
        idx = knn(x, k=k)

    # Prepare indices for gathering
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()

    # Gather neighbors for each point to construct local regions
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    # Expand x to match the dimensions for broadcasting subtraction
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # Concatenate the original point features with the relative positions to form the graph features
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class RegDGCNN(BaseModel):
    """
    Deep Graph Convolutional Neural Network for Regression Tasks (RegDGCNN) for processing 3D point cloud data.

    This network architecture extracts hierarchical features from point clouds using graph-based convolutions,
    enabling effective learning of spatial structures.
    """

    def __init__(self, args, output_channels=1):
        """
        Initializes the RegDGCNN model with specified configurations.

        Args:
            args (dict): Configuration parameters including 'k' for the number of neighbors, 'emb_dims' for embedding
            dimensions, and 'dropout' rate.
            output_channels (int): Number of output channels (e.g., for drag prediction, this is 1).
        """
        super(RegDGCNN, self).__init__()
        self.args = args
        self.k = args['k']  # Number of nearest neighbors
        self.batch_size = args["batch_size"]
        self.num_points = args["num_points"]

        # Batch normalization layers to stabilize and accelerate training
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(1024)
        self.bn5 = nn.BatchNorm1d(args['emb_dims'])

        # EdgeConv layers: Convolutional layers leveraging local neighborhood information
        self.conv1 = nn.Sequential(nn.Conv2d(14, 256, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(256 * 2, 512, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(512 * 2, 512, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(512 * 2, 1024, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(2304, args['emb_dims'], kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        # Fully connected layers to interpret the extracted features and make predictions
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

        self.linear5 = nn.Linear(16, output_channels)  # The final output layer

    @torch.autocast(device_type="cuda")
    def forward(self, x):
        """
        Forward pass of the model to process input data and predict outputs.

        Args:
            x (torch.Tensor): Input tensor representing a batch of point clouds.

        Returns:
            torch.Tensor: Model predictions for the input batch.
        """
        x=x.permute(0,2,1)
        batch_size = x.size(0)

        # Extract graph features and apply EdgeConv blocks
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 256, num_points, k)

        # Global max pooling
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        # Repeat the process for subsequent EdgeConv blocks
        x = get_graph_feature(x1, k=self.k)   # (batch_size, 256, num_points) -> (batch_size, 256*2, num_points, k)
        x = self.conv2(x)                     # (batch_size, 256*2, num_points, k) -> (batch_size, 512, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 512, num_points, k) -> (batch_size, 512, num_points)

        x = get_graph_feature(x2, k=self.k)   # (batch_size, 512, num_points) -> (batch_size, 512*2, num_points, k)
        x = self.conv3(x)                     # (batch_size, 512*2, num_points, k) -> (batch_size, 512, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 512, num_points, k) -> (batch_size, 512, num_points)

        x = get_graph_feature(x3, k=self.k)   # (batch_size, 512, num_points) -> (batch_size, 512*2, num_points, k)
        x = self.conv4(x)                     # (batch_size, 512*2, num_points, k) -> (batch_size, 1024, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points, k) -> (batch_size, 1024, num_points)

        # Concatenate features from all EdgeConv blocks
        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 256+512+512+1024, num_points)

        # Apply the final convolutional block
        x = self.conv5(x)  # (batch_size, 256+512+512+1024, num_points) -> (batch_size, emb_dims, num_points)
        # Combine global max and average pooling features
        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)   # (batch_size, emb_dims*2)

        # Process features through fully connected layers with dropout and batch normalization
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 128)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 128) -> (batch_size, 64)
        x = self.dp2(x)
        x = F.leaky_relu(self.bn8(self.linear3(x)), negative_slope=0.2)  # (batch_size, 64) -> (batch_size, 32)
        x = self.dp3(x)
        x = F.leaky_relu(self.bn9(self.linear4(x)), negative_slope=0.2)  # (batch_size, 32) -> (batch_size, 16)
        x = self.dp4(x)


        # x = F.leaky_relu(self.linear1(x), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 128)
        # x = self.dp1(x)
        # x = F.leaky_relu(self.linear2(x), negative_slope=0.2)  # (batch_size, 128) -> (batch_size, 64)
        # x = self.dp2(x)
        # x = F.leaky_relu(self.linear3(x), negative_slope=0.2)  # (batch_size, 64) -> (batch_size, 32)
        # x = self.dp3(x)
        # x = F.leaky_relu(self.linear4(x), negative_slope=0.2)  # (batch_size, 32) -> (batch_size, 16)
        # x = self.dp4(x)
        
        
        # Final linear layer to produce the output
        x = self.linear5(x)                                              # (batch_size, 16) -> (batch_size, 1)

        return x
    def data_dict_to_input(self, data_dict, **kwargs):
        if "vert_normals" in data_dict.keys():
            vert_normals = data_dict["vert_normals"].to(self.device)
            return vert_normals
        elif "cd" in data_dict.keys():
            # centroids = data_dict["centroids"][0].to(self.device)
            # areas = data_dict["areas"][0].unsqueeze(-1).to(self.device)
            # normal = data_dict["normal"][0].to(self.device)
            # ca = torch.cat([centroids, areas, normal], dim=1)
            # return ca
            ca=data_dict["cat_input"].to(self.device)
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
        vert_normal = self.data_dict_to_input(data_dict)
        # vert_normal = self._sample_or_pad_vertices(vert_normal, self.num_points)
        pred_var = self(vert_normal)
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
            # pred_var = torch.mean(pred_var).reshape(1,1)
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
        num_vertices = vertices.size(0)
        # Subsample the vertices if there are more than the desired number
        if num_vertices > num_points:
            indices = np.random.choice(num_vertices, num_points, replace=False)
            vertices = vertices[indices]
        # Pad with zeros if there are fewer vertices than desired
        elif num_vertices < num_points:
            padding = torch.zeros((num_points - num_vertices, 7), dtype=torch.float32).to(self.device)
            vertices = torch.cat((vertices, padding), dim=0)#仅仅是pad成0了
        return vertices

    def loss_dict(self, data_dict, loss_fn=None):
        vert_normal = self.data_dict_to_input(data_dict)
        # vert_normal = vert_normal[:: self.subsample_train]
        # vert_normal = self._sample_or_pad_vertices(vert_normal, self.num_points)#torch.Size([500000, 3])
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
            # pred_var = torch.mean(pred_var).reshape(1,1)
            true_var = data_dict["cd"]
        else:
            raise NotImplementedError("only pressure velocity works")

        return {"loss": loss_fn(pred_var.squeeze(-1), true_var.to(self.device))}
