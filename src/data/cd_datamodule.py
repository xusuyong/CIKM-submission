import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
sys.path.append("./PaddleScience/")
sys.path.append('/home/aistudio/3rd_lib')

from pathlib import Path
from src.data.base_datamodule import BaseDataModule
from src.data.velocity_datamodule import read, read_obj, centoirds


class CdDataset(Dataset):
    def __init__(self, input_dir, index_list, num_points):
        self.cd_list = np.loadtxt("../../xsy_datasets/CIKM_dataset/AIstudio_CIKM_dataset/Training/Dataset_2/Label_File/dataset2_train_label.csv", delimiter=",", dtype=str, encoding='utf-8')[:,2][1:].astype(np.float32)
        self.input_dir = input_dir
        self.index_list = index_list
        self.len = len(index_list)
        self.num_points=num_points
        
    def __getitem__(self, index):
        cd_label = self.cd_list[index]
        obj_name = self.index_list[index]
        data_dict = read(self.input_dir / f"{obj_name}.obj")
        
        
        centroids = data_dict["centroids"]
        areas = np.expand_dims(data_dict["areas"], axis=-1)
        normal = data_dict["normal"]
        cat_input = np.concatenate([centroids, areas, normal], axis=1)
        cat_input = self._sample_or_pad_vertices(torch.from_numpy(cat_input), self.num_points)
        data_dict={}
        data_dict["cd"] = cd_label
        data_dict["cat_input"] = cat_input
        return data_dict
    
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

    def __len__(self):
        return self.len


class CdDataModule(BaseDataModule):
    def __init__(self, train_data_dir, test_data_dir, train_index_list, test_index_list, num_points):
        BaseDataModule.__init__(self)
        self.train_data = CdDataset(Path(train_data_dir), train_index_list, num_points)
        self.test_data  = CdDataset(Path(test_data_dir),  test_index_list, num_points)
        self.train_indices = train_index_list
        self.test_indices = test_index_list
    
    def decode(self, x):
        return x
