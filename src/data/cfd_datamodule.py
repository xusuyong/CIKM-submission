from typing import List, Tuple, Union, Optional
from collections.abc import Callable
import warnings
import open3d as o3d
import numpy as np
from pathlib import Path
import unittest
import copy

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.base_datamodule import BaseDataModule
from neuralop.utils import UnitGaussianNormalizer


class DictDataset(Dataset):
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        for k, v in data_dict.items():
            assert len(v) == len(
                data_dict[list(data_dict.keys())[0]]
            ), "All data must have the same length"

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data_dict.items()}

    def __len__(self):
        return len(self.data_dict[list(self.data_dict.keys())[0]])


class DictDatasetWithConstant(DictDataset):
    def __init__(self, data_dict: dict, constant_dict: dict):
        super().__init__(data_dict)
        self.constant_dict = constant_dict

    def __getitem__(self, index):
        return_dict = {k: v[index] for k, v in self.data_dict.items()}
        return_dict.update(self.constant_dict)
        return return_dict


class VariableDictDataset(DictDataset):
    def __init__(
        self,
        data_dict: dict,
        path: str = None,
        location_norm: Optional[Callable] = None,
        pressure_norm: Optional[Callable] = None,
        info_norm: Optional[Callable] = None,
        area_norm: Optional[Callable] = None,
    ):
        DictDataset.__init__(self, data_dict)
        self.zfill = 4 if "data_train_B" in str(path) else 3
        self.path = Path(path) if path is not None else None
        self.location_norm = location_norm
        self.pressure_norm = pressure_norm
        self.info_norm = info_norm
        self.area_norm = area_norm

    def index_to_mesh_path(self, index, extension: str = ".ply") -> Path:
        return self.path / ("mesh_" + str(index).zfill(self.zfill) + extension)

    def index_to_pressure_path(self, index, extension: str = ".npy") -> Path:
        return self.path / ("press_" + str(index).zfill(3) + extension)

    def index_to_info_path(self, index, extension: str = ".pt") -> Path:
        return self.path / ("info_" + str(index).zfill(self.zfill) + extension)

    def load_mesh(self, mesh_path: Path) -> o3d.geometry.TriangleMesh:
        assert mesh_path.exists(), "Mesh path does not exist"
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        return mesh

    def load_pressure(self, pressure_path: Path) -> torch.Tensor:
        assert pressure_path.exists(), "Pressure path does not exist"
        pressure = np.load(str(pressure_path)).astype(np.float32)
        return torch.from_numpy(pressure)

    def load_info(self, info_path: Path) -> torch.Tensor:
        assert info_path.exists(), "Info path does not exist"
        info = torch.load(str(info_path))
        return info

    def vertices_from_mesh(self, mesh: o3d.geometry.TriangleMesh) -> torch.Tensor:
        return torch.from_numpy(np.asarray(mesh.vertices).astype(np.float32))

    def triangles_from_mesh(self, mesh: o3d.geometry.TriangleMesh) -> torch.Tensor:
        return torch.from_numpy(np.asarray(mesh.triangles).astype(np.int64))

    def get_triangle_centroids(
        self, vertices: torch.Tensor, triangles: torch.Tensor
    ) -> torch.Tensor:
        A, B, C = (
            vertices[triangles[:, 0]],
            vertices[triangles[:, 1]],
            vertices[triangles[:, 2]],
        )

        centroids = (A + B + C) / 3
        areas = torch.sqrt(torch.sum(torch.cross(B - A, C - A) ** 2, 1)) / 2

        return centroids, areas

    def __getitem__(self, index):
        return_dict = {k: v[index] for k, v in self.data_dict.items()}

        if self.path is not None:
            mesh = self.load_mesh(self.index_to_mesh_path(index + 1))
            info = self.load_info(self.index_to_info_path(index + 1))

            vertices = self.vertices_from_mesh(mesh)
            triangles = self.triangles_from_mesh(mesh)
            centroids, areas = self.get_triangle_centroids(vertices, triangles)

            pressure = self.load_pressure(self.index_to_pressure_path(index + 1))

            if self.location_norm is not None:
                vertices = self.location_norm(vertices)
                centroids = self.location_norm(centroids)

            if self.pressure_norm is not None:
                pressure = self.pressure_norm(pressure)

            if self.info_norm is not None:
                info = self.info_norm(info)

            if self.area_norm is not None:
                areas = self.area_norm(areas)

            return_dict["vertices"] = vertices
            return_dict["centroids"] = centroids
            return_dict["areas"] = areas
            return_dict["pressure"] = pressure
            return_dict["info"] = info

        return return_dict


class VariableDictDatasetWithConstant(VariableDictDataset):
    def __init__(
        self,
        data_dict: dict,
        constant_dict: dict,
        path: str = None,
        localtion_norm: Optional[Callable] = None,
        pressure_norm: Optional[Callable] = None,
        info_norm: Optional[Callable] = None,
        area_norm: Optional[Callable] = None,
    ):
        super().__init__(
            data_dict, path, localtion_norm, pressure_norm, info_norm, area_norm
        )
        self.constant_dict = constant_dict

    def __getitem__(self, index):
        return_dict = super().__getitem__(index)
        return_dict.update(self.constant_dict)
        return return_dict


class VariableDictDatasetTrackB(DictDataset):
    def __init__(
        self,
        data_dict: dict,
        path: str = None,
        location_norm: Optional[Callable] = None,
        pressure_norm: Optional[Callable] = None,
        # info_norm: Optional[Callable] = None,
        area_norm: Optional[Callable] = None,
    ):
        DictDataset.__init__(self, data_dict)
        if (
            "data_train_B" in str(path)
            or "data_test_B" in str(path)
            or "data_validate_B" in str(path)
        ):
            self.zfill = 4
        else:
            self.zfill = 3
        self.path = Path(path) if path is not None else None
        self.location_norm = location_norm
        self.pressure_norm = pressure_norm
        # self.info_norm = info_norm
        self.area_norm = area_norm
        # data_module = self.constant_dict["data_module"]
        # train_pressure_lst=data_module.train_pressure_lst

    def index_to_mesh_path(self, index, extension: str = ".ply") -> Path:
        return self.path / ("mesh_" + str(index).zfill(self.zfill) + extension)

    def index_to_pressure_path(self, index, extension: str = ".npy") -> Path:
        return self.path / ("press_" + str(index).zfill(self.zfill) + extension)

    def index_to_centroids_path(self, index, extension: str = ".npy") -> Path:
        return self.path / ("centroid_" + str(index).zfill(self.zfill) + extension)

    def index_to_areas_path(self, index, extension: str = ".npy") -> Path:
        return self.path / ("area_" + str(index).zfill(self.zfill) + extension)

    # def index_to_info_path(self, index, extension: str = ".pt") -> Path:
    #     return self.path / ("info_" + str(index).zfill(self.zfill) + extension)

    def load_mesh(self, mesh_path: Path) -> o3d.geometry.TriangleMesh:
        assert mesh_path.exists(), "Mesh path does not exist"
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        return mesh

    def load_pressure(self, pressure_path: Path) -> torch.Tensor:
        # print(pressure_path)
        assert pressure_path.exists(), "Pressure path does not exist"
        pressure = np.load(str(pressure_path)).astype(np.float32)
        return torch.from_numpy(pressure)

    def load_centroids(self, centroids_path: Path) -> torch.Tensor:
        # print(centroids_path)
        assert centroids_path.exists(), "centroids path does not exist"
        centroids = np.load(str(centroids_path)).astype(np.float32)
        return torch.from_numpy(centroids)

    def load_areas(self, areas_path: Path) -> torch.Tensor:
        # print(areas_path)
        assert areas_path.exists(), "areas path does not exist"
        areas = np.load(str(areas_path)).astype(np.float32)
        return torch.from_numpy(areas)

    def load_info(self, info_path: Path) -> torch.Tensor:
        assert info_path.exists(), "Info path does not exist"
        info = torch.load(str(info_path))
        return info

    def vertices_from_mesh(self, mesh: o3d.geometry.TriangleMesh) -> torch.Tensor:
        return torch.from_numpy(np.asarray(mesh.vertices).astype(np.float32))

    def triangles_from_mesh(self, mesh: o3d.geometry.TriangleMesh) -> torch.Tensor:
        return torch.from_numpy(np.asarray(mesh.triangles).astype(np.int64))

    def get_triangle_centroids(
        self, vertices: torch.Tensor, triangles: torch.Tensor
    ) -> torch.Tensor:
        A, B, C = (
            vertices[triangles[:, 0]],
            vertices[triangles[:, 1]],
            vertices[triangles[:, 2]],
        )

        centroids = (A + B + C) / 3
        areas = torch.sqrt(torch.sum(torch.cross(B - A, C - A) ** 2, 1)) / 2

        return centroids, areas

    def __getitem__(self, index):
        return_dict = {k: v[index] for k, v in self.data_dict.items()}

        if self.path is not None:
            # mesh = self.load_mesh(self.index_to_mesh_path(index + 1))
            # info = self.load_info(self.index_to_info_path(index + 1))

            # vertices = self.vertices_from_mesh(mesh)
            # triangles = self.triangles_from_mesh(mesh)
            # centroids, areas = self.get_triangle_centroids(vertices, triangles)

            # pressure=train_pressure_lst[index]
            # p = data_module.load_pressure(data_dir, "", mesh_index)
            # p = paddle.to_tensor(data=p, dtype="float32")

            # encode [p]
            # encode = data_module.pressure_normalization.encode
            # p = encode(p)
            # return_dict["pressure"] = p
            pressure = self.load_pressure(
                self.index_to_pressure_path(self.indices[index])
            )
            centroids = self.load_centroids(
                self.index_to_centroids_path(self.indices[index])
            )
            areas = self.load_areas(self.index_to_areas_path(self.indices[index]))
            # TODO 检查这些归一化是否匹配
            if self.location_norm is not None:
                # vertices = self.location_norm(vertices)
                centroids = self.location_norm(centroids)

            if self.pressure_norm is not None:
                pressure = self.pressure_norm(pressure)

            # if self.info_norm is not None:
            #     info = self.info_norm(info)

            if self.area_norm is not None:
                areas = self.area_norm(areas)

            # return_dict["vertices"] = vertices
            # return_dict["centroids"] = centroids
            # return_dict["areas"] = areas
            # 直接加载文件
            return_dict["pressure"] = pressure
            # return_dict["info"] = info
            return_dict["centroids"] = centroids
            return_dict["areas"] = areas

        return return_dict


class VariableDictDatasetTrackBWithConstant(VariableDictDatasetTrackB):
    def __init__(
        self,
        data_dict: dict,
        constant_dict: dict,
        path: str = None,
        localtion_norm: Optional[Callable] = None,
        pressure_norm: Optional[Callable] = None,
        # info_norm: Optional[Callable] = None,
        area_norm: Optional[Callable] = None,
        indices=None,
    ):
        super().__init__(data_dict, path, localtion_norm, pressure_norm, area_norm)
        self.constant_dict = constant_dict
        self.indices = indices

    def __getitem__(self, index):
        return_dict = super().__getitem__(index)
        return_dict.update(self.constant_dict)

        return return_dict


class CFDDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir,
        n_train: int = 500,
        n_test: int = 111,
    ):
        super().__init__()

        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        data_dir = data_dir.expanduser()
        assert data_dir.exists(), "Path does not exist"
        assert data_dir.is_dir(), "Path is not a directory"
        self.data_dir = data_dir

        valid_mesh_inds = self.load_valid_mesh_indices(data_dir)
        assert n_train + n_test <= len(valid_mesh_inds), "Not enough data"
        if (n_train + n_test) < len(valid_mesh_inds):
            warnings.warn(
                f"{len(valid_mesh_inds)} meshes are available, but {n_train + n_test} are requested."
            )
        # split the valid_mesh_inds into n_train and n_test indices and use the indices to load the mesh

        train_indices = valid_mesh_inds[:n_train]
        # test_indices = valid_mesh_inds[-n_test:]
        test_indices = valid_mesh_inds[n_train : n_train + n_test]
        train_mesh_pathes = [self.get_mesh_path(data_dir, i) for i in train_indices]
        test_mesh_pathes = [self.get_mesh_path(data_dir, i) for i in test_indices]
        self.train_mesh_pathes = train_mesh_pathes
        self.test_mesh_pathes = test_mesh_pathes
        infer_data_dir = Path("../../xsy_datasets/IJCAI_dataset/aistuio/data_test_A")
        infer_data_dir = infer_data_dir.expanduser()
        assert infer_data_dir.exists(), "Path does not exist"
        assert infer_data_dir.is_dir(), "Path is not a directory"
        self.infer_data_dir = infer_data_dir
        infer_valid_mesh_inds = self.load_valid_mesh_indices(infer_data_dir)
        self.infer_mesh_pathes = [
            self.get_infer_mesh_path(infer_data_dir, i) for i in infer_valid_mesh_inds
        ]
        train_vertices = [
            self.vertices_from_mesh(mesh_path) for mesh_path in train_mesh_pathes
        ]  # 500个(3586, 3)
        test_vertices = [
            self.vertices_from_mesh(mesh_path) for mesh_path in test_mesh_pathes
        ]  # 111个(3586, 3)
        self.infer_vertices = [
            self.vertices_from_mesh(mesh_path) for mesh_path in self.infer_mesh_pathes
        ]  # 50个(3586, 3)

        train_pressure = torch.stack(
            [
                torch.Tensor(self.load_pressure(data_dir, mesh_index))
                for mesh_index in train_indices
            ]
        )  # torch.Size([500, 3586])
        test_pressure = torch.stack(
            [
                torch.Tensor(self.load_pressure(data_dir, mesh_index))
                for mesh_index in test_indices
            ]
        )  # torch.Size([111, 3586])
        # normalize pressure
        pressure_normalization = UnitGaussianNormalizer(
            train_pressure, eps=1e-6, reduce_dim=[0, 1], verbose=False
        )
        train_pressure = pressure_normalization.encode(train_pressure)
        test_pressure = pressure_normalization.encode(test_pressure)

        self._train_data = DictDataset(
            {
                "vertices": train_vertices,
                "pressure": train_pressure,
            },
        )
        self._test_data = DictDataset(
            {
                "vertices": test_vertices,
                "pressure": test_pressure,
            },
        )

        self.output_normalization = pressure_normalization

    def encode(self, pressure: torch.Tensor) -> torch.Tensor:
        self.output_normalization.to(pressure.device)
        return self.output_normalization.encode(pressure)

    def decode(self, pressure: torch.Tensor) -> torch.Tensor:
        self.output_normalization.to(pressure.device)
        return self.output_normalization.decode(pressure)

    @property
    def train_data(self):
        return self._train_data

    @property
    def test_data(self):
        return self._test_data

    @property
    def infer_data(self):
        return self._infer_data

    def vertices_from_mesh(self, mesh_path: Path) -> torch.Tensor:
        mesh = self.load_mesh(mesh_path)
        vertices = mesh.vertex.positions.numpy()
        return vertices

    def get_mesh_path(self, data_dir: Path, mesh_ind: int) -> Path:
        return data_dir / "data" / ("mesh_" + str(mesh_ind).zfill(3) + ".ply")

    def get_infer_mesh_path(self, data_dir: Path, mesh_ind: int) -> Path:
        return data_dir / ("mesh_" + str(mesh_ind).zfill(3) + ".ply")

    def get_pressure_data_path(self, data_dir: Path, mesh_ind: int) -> Path:
        return data_dir / "data" / ("press_" + str(mesh_ind).zfill(3) + ".npy")

    def load_pressure(self, data_dir: Path, mesh_index: int) -> np.ndarray:
        press_path = self.get_pressure_data_path(data_dir, mesh_index)
        # print(press_path)
        assert press_path.exists(), "Pressure data does not exist"
        press = np.load(press_path).reshape((-1,)).astype(np.float32)  # (3682,)
        press = np.concatenate(
            (press[0:16], press[112:]), axis=0
        )  # (3586,) 因为被去掉的点不是表面点
        return press

    def load_valid_mesh_indices(
        self, data_dir, filename="watertight_meshes.txt"
    ) -> List[int]:
        with open(data_dir / filename, "r") as fp:
            mesh_ind = fp.read().split("\n")
            mesh_ind = [int(a) for a in mesh_ind]
        return mesh_ind

    def load_mesh(self, mesh_path: Path) -> o3d.t.geometry.TriangleMesh:
        # print(mesh_path)
        assert mesh_path.exists(), "Mesh path does not exist"
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        return mesh

    def load_mesh_from_index(
        self, data_dir, mesh_index: int
    ) -> o3d.t.geometry.TriangleMesh:
        mesh_path = self.get_mesh_path(data_dir, mesh_index)
        return self.load_mesh(mesh_path)


class CFDSDFDataModule(CFDDataModule):
    def __init__(
        self,
        data_dir,
        n_train: int = 500,
        n_test: int = 111,
        spatial_resolution: Tuple[int, int, int] = None,
        query_points=None,
        eps=1e-2,
        closest_points_to_query=True,
    ):
        BaseDataModule.__init__(self)

        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        data_dir = data_dir.expanduser()
        assert data_dir.exists(), "Path does not exist"
        assert data_dir.is_dir(), "Path is not a directory"
        self.data_dir = data_dir

        min_bounds, max_bounds = self.load_bound(data_dir, eps=eps)
        valid_mesh_inds = self.load_valid_mesh_indices(data_dir)
        assert n_train + n_test <= len(valid_mesh_inds), "Not enough data"
        if (n_train + n_test) < len(valid_mesh_inds):
            warnings.warn(
                f"{len(valid_mesh_inds)} meshes are available, but {n_train + n_test} are requested."
            )
        # split the valid_mesh_inds into n_train and n_test indices and use the indices to load the mesh

        train_indices = valid_mesh_inds[:n_train]
        # test_indices = valid_mesh_inds[-n_test:]
        test_indices = valid_mesh_inds[n_train : n_train + n_test]
        train_mesh_pathes = [self.get_mesh_path(data_dir, i) for i in train_indices]
        test_mesh_pathes = [self.get_mesh_path(data_dir, i) for i in test_indices]

        infer_data_dir = Path("../../xsy_datasets/IJCAI_dataset/aistuio/data_test_A")
        infer_data_dir = infer_data_dir.expanduser()
        assert infer_data_dir.exists(), "Path does not exist"
        assert infer_data_dir.is_dir(), "Path is not a directory"
        self.infer_data_dir = infer_data_dir
        self.infer_valid_mesh_inds = self.load_valid_mesh_indices(infer_data_dir)
        self.infer_mesh_pathes = [
            self.get_infer_mesh_path(infer_data_dir, i) for i in self.infer_valid_mesh_inds
        ]

        if query_points is None:
            assert spatial_resolution is not None, "spatial_resolution must be given"
            tx = np.linspace(min_bounds[0], max_bounds[0], spatial_resolution[0])
            ty = np.linspace(min_bounds[1], max_bounds[1], spatial_resolution[1])
            tz = np.linspace(min_bounds[2], max_bounds[2], spatial_resolution[2])
            query_points = np.stack(
                np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            ).astype(
                np.float32
            )  # (32, 32, 32, 3) 3表示x,y,z

        train_sdf_mesh_vertices = [
            self.sdf_vertices_closest_from_mesh(
                mesh_path, query_points, closest_points_to_query
            )
            for mesh_path in train_mesh_pathes
        ]
        train_sdf = torch.stack(
            [torch.Tensor(sdf) for sdf, _, _ in train_sdf_mesh_vertices]
        )  # torch.Size([500, 32, 32, 32])
        train_vertices = torch.stack(
            [torch.Tensor(vertices) for _, vertices, _ in train_sdf_mesh_vertices]
        )
        if closest_points_to_query:
            train_closest_points = torch.stack(
                [torch.Tensor(closest) for _, _, closest in train_sdf_mesh_vertices]
            )
        else:
            train_closest_points = None

        del train_sdf_mesh_vertices
        train_pressure = torch.stack(
            [
                torch.Tensor(self.load_pressure(data_dir, mesh_index))
                for mesh_index in train_indices
            ]
        )

        # Location normalization
        min_bounds = torch.tensor(min_bounds)
        max_bounds = torch.tensor(max_bounds)

        # normalize train_vertices with min_bounds max_bounds
        train_vertices = self.location_normalization(
            train_vertices, min_bounds, max_bounds
        )

        if closest_points_to_query:
            train_closest_points = self.location_normalization(
                train_closest_points, min_bounds, max_bounds
            ).permute(0, 4, 1, 2, 3)

        test_sdf_mesh_vertices = [
            self.sdf_vertices_closest_from_mesh(
                mesh_path, query_points, closest_points_to_query
            )
            for mesh_path in test_mesh_pathes
        ]

        infer_sdf_mesh_vertices = [
            self.sdf_vertices_closest_from_mesh(
                mesh_path, query_points, closest_points_to_query
            )
            for mesh_path in self.infer_mesh_pathes
        ]
        test_sdf = torch.stack(
            [torch.Tensor(sdf) for sdf, _, _ in test_sdf_mesh_vertices]
        )
        infer_sdf = torch.stack(
            [torch.Tensor(sdf) for sdf, _, _ in infer_sdf_mesh_vertices]
        )
        test_vertices = torch.stack(
            [torch.Tensor(vertices) for _, vertices, _ in test_sdf_mesh_vertices]
        )
        infer_vertices = torch.stack(
            [torch.Tensor(vertices) for _, vertices, _ in infer_sdf_mesh_vertices]
        )
        if closest_points_to_query:
            test_closest_points = torch.stack(
                [torch.Tensor(closest) for _, _, closest in test_sdf_mesh_vertices]
            )
        else:
            test_closest_points = None

        del test_sdf_mesh_vertices
        test_pressure = torch.stack(
            [
                torch.Tensor(self.load_pressure(data_dir, mesh_index))
                for mesh_index in test_indices
            ]
        )
        test_vertices = self.location_normalization(
            test_vertices, min_bounds, max_bounds
        )
        infer_vertices = self.location_normalization(
            infer_vertices, min_bounds, max_bounds
        )
        if closest_points_to_query:
            test_closest_points = self.location_normalization(
                test_closest_points, min_bounds, max_bounds
            ).permute(0, 4, 1, 2, 3)

        # normalize pressure
        pressure_normalization = UnitGaussianNormalizer(
            train_pressure, eps=1e-6, reduce_dim=[0, 1], verbose=False
        )
        train_pressure = pressure_normalization.encode(train_pressure)
        test_pressure = pressure_normalization.encode(test_pressure)

        # normalize query points
        normalized_query_points = self.location_normalization(
            torch.from_numpy(query_points), min_bounds, max_bounds
        ).permute(3, 0, 1, 2)

        self._train_data = DictDatasetWithConstant(
            {
                "sdf": train_sdf,
                "vertices": train_vertices,
                "pressure": train_pressure,
            },
            {"sdf_query_points": normalized_query_points},
        )
        self._test_data = DictDatasetWithConstant(
            {
                "sdf": test_sdf,
                "vertices": test_vertices,
                "pressure": test_pressure,
            },
            {"sdf_query_points": normalized_query_points},
        )
        self._infer_data = DictDatasetWithConstant(
            {
                "sdf": infer_sdf,
                "vertices": infer_vertices,
            },
            {"sdf_query_points": normalized_query_points},
        )

        if closest_points_to_query:
            self._train_data.data_dict["closest_points"] = (
                train_closest_points  # torch.Size([500, 3, 64, 64, 64])
            )
            self._test_data.data_dict["closest_points"] = test_closest_points

        self.output_normalization = pressure_normalization

    def load_bound(
        self, data_dir, filename="watertight_global_bounds.txt", eps=1e-6
    ) -> Tuple[List[float], List[float]]:
        with open(data_dir / filename, "r") as fp:
            min_bounds = fp.readline().split(" ")
            max_bounds = fp.readline().split(" ")

            min_bounds = [float(a) - eps for a in min_bounds]
            max_bounds = [float(a) + eps for a in max_bounds]
        return (
            min_bounds,
            max_bounds,
        )  # 确保在边界值附近略微扩展一些余地，从而避免由于计算精度问题导致的边界误差。

    def location_normalization(
        self,
        locations: torch.Tensor,
        min_bounds: torch.Tensor,
        max_bounds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize locations to [-1, 1].
        """
        locations = (locations - min_bounds) / (max_bounds - min_bounds)
        locations = 2 * locations - 1
        return locations

    def compute_sdf(
        self, mesh: Union[Path, o3d.t.geometry.TriangleMesh], query_points
    ) -> np.ndarray:
        if isinstance(mesh, Path):
            mesh = self.load_mesh(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)  # 添加网格到场景中
        signed_distance = scene.compute_signed_distance(
            query_points
        ).numpy()  # (32, 32, 32) 核心计算
        return signed_distance

    # 从一个三角形网格（TriangleMesh）中找到距离查询点（query_points）最近的点
    def closest_points_to_query_from_mesh(
        self, mesh: o3d.t.geometry.TriangleMesh, query_points
    ) -> np.ndarray:
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        # 注意不是从vertices上找而是直接从mesh上找所以closest_points每个点都是唯一的
        closest_points = scene.compute_closest_points(query_points)["points"].numpy()

        return closest_points

    def sdf_vertices_closest_from_mesh(
        self,
        mesh_path: Path,
        query_points: np.ndarray,
        closest_points: bool,
    ) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        mesh = self.load_mesh(mesh_path)
        sdf = self.compute_sdf(mesh, query_points)  # sdf就是在这里计算的 (32, 32, 32)
        vertices = mesh.vertex.positions.numpy()  # (3586, 3)
        if closest_points:
            closest_points = self.closest_points_to_query_from_mesh(
                mesh, query_points
            )  # (32, 32, 32, 3)
        else:
            closest_points = None

        return sdf, vertices, closest_points


class CFDNormalDataModule(CFDDataModule):
    def __init__(
        self,
        data_dir,
        n_train: int = 500,
        n_test: int = 111,
        eps=1e-2,
        use_multifi=False,
    ):
        # super().__init__(data_dir, n_train, n_test)
        BaseDataModule.__init__(self)

        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        data_dir = data_dir.expanduser()
        assert data_dir.exists(), "Path does not exist"
        assert data_dir.is_dir(), "Path is not a directory"
        self.data_dir = data_dir

        min_bounds, max_bounds = self.load_bound(data_dir, eps=eps)
        valid_mesh_inds = self.load_valid_mesh_indices(data_dir)
        assert n_train + n_test <= len(valid_mesh_inds), "Not enough data"
        if (n_train + n_test) < len(valid_mesh_inds):
            warnings.warn(
                f"{len(valid_mesh_inds)} meshes are available, but {n_train + n_test} are requested."
            )
        # split the valid_mesh_inds into n_train and n_test indices and use the indices to load the mesh

        train_indices = valid_mesh_inds[:n_train]
        # test_indices = valid_mesh_inds[-n_test:]
        test_indices = valid_mesh_inds[n_train : n_train + n_test]
        train_mesh_pathes = [self.get_mesh_path(data_dir, i) for i in train_indices]
        test_mesh_pathes = [self.get_mesh_path(data_dir, i) for i in test_indices]
        self.train_indices = train_indices
        self.test_indices = test_indices

        infer_data_dir = Path("../../xsy_datasets/IJCAI_dataset/aistuio/data_test_A")
        infer_data_dir = infer_data_dir.expanduser()
        assert infer_data_dir.exists(), "Path does not exist"
        assert infer_data_dir.is_dir(), "Path is not a directory"
        self.infer_data_dir = infer_data_dir
        infer_valid_mesh_inds = self.load_valid_mesh_indices(infer_data_dir)
        self.infer_mesh_pathes = [
            self.get_infer_mesh_path(infer_data_dir, i) for i in infer_valid_mesh_inds
        ]

        # train_normals=self.comput_normal(self.mesh_path)
        # test_normals=self.comput_normal(self.mesh_path)
        # infer_normals=self.comput_normal(self.mesh_path)
        train_vertices_normals = [
            self.comput_geom(mesh_path) for mesh_path in train_mesh_pathes
        ]
        test_vertices_normals = [
            self.comput_geom(mesh_path) for mesh_path in test_mesh_pathes
        ]
        infer_vertices_normals = [
            self.comput_geom(mesh_path) for mesh_path in self.infer_mesh_pathes
        ]

        train_vertices = torch.stack(
            [torch.Tensor(vertices) for vertices, _ in train_vertices_normals]
        )
        test_vertices = torch.stack(
            [torch.Tensor(vertices) for vertices, _ in test_vertices_normals]
        )
        infer_vertices = torch.stack(
            [torch.Tensor(vertices) for vertices, _ in infer_vertices_normals]
        )

        train_normals = torch.stack(
            [torch.Tensor(normals) for _, normals in train_vertices_normals]
        )
        test_normals = torch.stack(
            [torch.Tensor(normals) for _, normals in test_vertices_normals]
        )
        infer_normals = torch.stack(
            [torch.Tensor(normals) for _, normals in infer_vertices_normals]
        )

        del train_vertices_normals
        train_pressure = torch.stack(
            [
                torch.Tensor(self.load_pressure(data_dir, mesh_index))
                for mesh_index in train_indices
            ]
        )
        del test_vertices_normals
        test_pressure = torch.stack(
            [
                torch.Tensor(self.load_pressure(data_dir, mesh_index))
                for mesh_index in test_indices
            ]
        )
        del infer_vertices_normals

        # Location normalization
        min_bounds = torch.tensor(min_bounds)
        max_bounds = torch.tensor(max_bounds)

        # normalize train_vertices with min_bounds max_bounds
        train_vertices = self.location_normalization(
            train_vertices, min_bounds, max_bounds
        )
        test_vertices = self.location_normalization(
            test_vertices, min_bounds, max_bounds
        )
        infer_vertices = self.location_normalization(
            infer_vertices, min_bounds, max_bounds
        )

        if use_multifi:
            press_pred_train_dir = Path("logs/2024-06-29_20-19-58_trackA/Track_A_pred_train")
            press_pred_test_dir = Path("logs/2024-06-29_20-19-58_trackA/Track_A_pred_test")
            pressure_pred_train = torch.stack(
                [
                    torch.Tensor(self.load_pressure(press_pred_train_dir, mesh_index))
                    for mesh_index in train_indices
                ]
            )
            pressure_pred_test = torch.stack(
                [
                    torch.Tensor(self.load_pressure(press_pred_test_dir, mesh_index))
                    for mesh_index in test_indices
                ]
            )
        else:
            pressure_pred_train = torch.zeros_like(train_pressure)
            pressure_pred_test = torch.zeros_like(test_pressure)
            infer_pressure = test_pressure
            pressure_pred_infer = torch.zeros_like(infer_pressure)

        # normalize pressure
        # pressure_normalization = UnitGaussianNormalizer(train_pressure, eps=1e-6, reduce_dim=[0, 1], verbose=False)
        pressure_normalization=None
        # train_pressure = pressure_normalization.encode(train_pressure)
        # test_pressure = pressure_normalization.encode(test_pressure)

        if use_multifi:
            pressure_pred_train = pressure_normalization.encode(pressure_pred_train)
            pressure_pred_test = pressure_normalization.encode(pressure_pred_test)
            pressure_pred_infer = pressure_normalization.encode(pressure_pred_infer)

        train_vert_normals = torch.cat(
            (train_vertices, train_normals), dim=2
        )  # torch.Size([500, 3586, 6])
        test_vert_normals = torch.cat(
            (test_vertices, test_normals), dim=2
        )  # torch.Size([50, 3586, 6])
        infer_vert_normals = torch.cat(
            (infer_vertices, infer_normals), dim=2
        )  # torch.Size([50, 3586, 6])
        # torch.save(train_pressure, 'train_pressure.pth')
        # torch.save(test_pressure, 'test_pressure.pth')
        # torch.save(train_vert_normals, 'train_vert_normals.pth')
        # torch.save(test_vert_normals, 'test_vert_normals.pth')
        # torch.save(infer_vert_normals, 'infer_vert_normals.pth')
        self._train_data = DictDataset(
            {
                # "vertices": self.train_vertices,
                "pressure": train_pressure,
                "vert_normals": train_vert_normals,
                "pressure_pred": pressure_pred_train,
            },
        )
        self._test_data = DictDataset(
            {
                # "vertices": self.test_vertices,
                "pressure": test_pressure,
                "vert_normals": test_vert_normals,
                "pressure_pred": pressure_pred_test,
            },
        )
        self._infer_data = DictDataset(
            {
                # "vertices": self.infer_vertices,
                # "pressure": test_pressure,
                "vert_normals": infer_vert_normals,
                # "pressure_pred": pressure_pred_infer,
            },
        )

        self.output_normalization = pressure_normalization

    def comput_geom(self, mesh_path):
        # mesh_path = Path(mesh_path)
        assert mesh_path.exists(), "Mesh path does not exist"

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()  # 计算顶点法线

        # 将法线转换为NumPy数组
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)  # (N, 3)

        # 转换为新的TriangleMesh格式
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        vertices = mesh.vertex.positions.numpy()  # (N, 3)

        # 确保 vertices 和 normals 的形状相同
        assert (
            vertices.shape == normals.shape
        ), "Shape mismatch between vertices and normals"

        # 合并顶点和法线数据
        # vertices_normals = np.concatenate((vertices, normals), axis=1)  # 合并在 axis=1 处

        return vertices, normals

    def load_bound(
        self, data_dir, filename="watertight_global_bounds.txt", eps=1e-6
    ) -> Tuple[List[float], List[float]]:
        with open(data_dir / filename, "r") as fp:
            min_bounds = fp.readline().split(" ")
            max_bounds = fp.readline().split(" ")

            min_bounds = [float(a) - eps for a in min_bounds]
            max_bounds = [float(a) + eps for a in max_bounds]
        return (
            min_bounds,
            max_bounds,
        )  # 确保在边界值附近略微扩展一些余地，从而避免由于计算精度问题导致的边界误差。

    def location_normalization(
        self,
        locations: torch.Tensor,
        min_bounds: torch.Tensor,
        max_bounds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize locations to [-1, 1].
        """
        locations = (locations - min_bounds) / (max_bounds - min_bounds)
        locations = 2 * locations - 1
        return locations


class TrackBDataModule(CFDSDFDataModule):
    def __init__(
        self,
        data_dir,
        test_data_dir,
        n_train: int = 1,
        n_test: int = 1,
        spatial_resolution: Tuple[int, int, int] = None,
        query_points=None,
        eps=0.01,
        closest_points_to_query=True,
        Require_sdf=True,
    ):
        BaseDataModule.__init__(self)
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
            test_data_dir = Path(test_data_dir)
        data_dir = data_dir.expanduser()
        test_data_dir = test_data_dir.expanduser()
        print(data_dir)
        print(test_data_dir)
        assert data_dir.exists(), "Path does not exist"
        assert data_dir.is_dir(), "Path is not a directory"
        self.data_dir = data_dir
        self.test_data_dir = test_data_dir
        min_bounds, max_bounds = self.load_bound(
            data_dir, filename="global_bounds.txt", eps=eps
        )  # 3
        min_info_bounds, max_info_bounds = self.load_bound(
            test_data_dir, filename="info_bounds.txt", eps=0.0
        )  # 8
        min_area_bound, max_area_bound = self.load_bound(
            test_data_dir, filename="area_bounds.txt", eps=0.0
        )
        assert n_train <= 500, "Not enough training data"
        assert n_test <= 51, "Not enough testing data"
        if n_train + n_test < 551:
            warnings.warn(
                f"551 meshes are available, but {n_train + n_test} are requested."
            )
        train_indices = np.loadtxt(data_dir / "train_index.txt", dtype=int)
        test_indices_fake = train_indices
        train_indices = train_indices[:n_train]
        if "data_validate_B" in str(test_data_dir):
            test_indices = test_indices_fake[-n_test:]
        else:
            test_indices = [(j + 1) for j in range(n_test)]
        self.test_indices = test_indices

        if query_points is None:
            assert spatial_resolution is not None, "spatial_resolution must be given"
            tx = np.linspace(min_bounds[0], max_bounds[0], spatial_resolution[0])
            ty = np.linspace(min_bounds[1], max_bounds[1], spatial_resolution[1])
            tz = np.linspace(min_bounds[2], max_bounds[2], spatial_resolution[2])
            query_points = np.stack(
                np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            ).astype(
                np.float32
            )  # (64, 64, 64, 3)

        if Require_sdf:
            train_mesh_pathes = [
                self.get_mesh_path(data_dir, "", i) for i in train_indices
            ]
            test_mesh_pathes = [
                self.get_mesh_path(test_data_dir, "", i) for i in test_indices
            ]
            self.test_mesh_pathes = test_mesh_pathes

            train_df_closest = [
                self.df_from_mesh(mesh_path, query_points, closest_points_to_query)
                for mesh_path in train_mesh_pathes
            ]
            train_df = torch.stack(
                [torch.Tensor(df) for df, _ in train_df_closest]
            )  # torch.Size([m, 64, 64, 64])
            if closest_points_to_query:
                train_closest_points = torch.stack(
                    [torch.Tensor(closest) for _, closest in train_df_closest]
                )  # torch.Size([m, 64, 64, 64, 3])
            else:
                train_closest_points = None

            del train_df_closest

            test_df_closest = [
                self.df_from_mesh(mesh_path, query_points, closest_points_to_query)
                for mesh_path in test_mesh_pathes
            ]
            test_df = torch.stack(
                [torch.Tensor(df) for df, _ in test_df_closest]
            )  # torch.Size([B, 64, 64, 64])
            if closest_points_to_query:
                test_closest_points = torch.stack(
                    [torch.Tensor(closest) for _, closest in test_df_closest]
                )  # torch.Size([B, 64, 64, 64, 3])
            else:
                test_closest_points = None

            del test_df_closest

            min_bounds = torch.tensor(min_bounds)
            max_bounds = torch.tensor(max_bounds)

            normalized_query_points = self.location_normalization(
                torch.from_numpy(query_points), min_bounds, max_bounds
            ).permute(3, 0, 1, 2)

            if closest_points_to_query:
                train_closest_points = self.location_normalization(
                    train_closest_points, min_bounds, max_bounds
                ).permute(0, 4, 1, 2, 3)

                test_closest_points = self.location_normalization(
                    test_closest_points, min_bounds, max_bounds
                ).permute(0, 4, 1, 2, 3)
        else:
            train_df = torch.zeros([n_train, 64, 64, 64])
            test_df = torch.zeros([n_test, 64, 64, 64])

            min_bounds = torch.tensor(min_bounds)
            max_bounds = torch.tensor(max_bounds)
            normalized_query_points = self.location_normalization(
                torch.from_numpy(query_points), min_bounds, max_bounds
            ).permute(3, 0, 1, 2)

            if closest_points_to_query:
                train_closest_points = self.location_normalization(
                    train_closest_points, min_bounds, max_bounds
                ).permute(0, 4, 1, 2, 3)

                test_closest_points = self.location_normalization(
                    test_closest_points, min_bounds, max_bounds
                ).permute(0, 4, 1, 2, 3)

        # normalize pressure
        train_pressure_lst = [
            torch.from_numpy(self.load_pressure(data_dir, "", i)) for i in train_indices
        ]

        train_pressure = torch.cat(
            train_pressure_lst
        )  # torch.Size([774500])=((403016,),)和((371484,),)竖向堆叠

        pressure_normalization = UnitGaussianNormalizer(
            train_pressure, eps=1e-6, reduce_dim=[0], verbose=False
        )

        location_norm_fn = lambda x: self.location_normalization(x, min_bounds, max_bounds)
        # location_norm_fn=None

        area_norm_fn = lambda x: self.area_normalization(x, min_area_bound[0], max_area_bound[0])
        # area_norm_fn=None
        
        self._train_data = VariableDictDatasetTrackBWithConstant(
            data_dict={
                "sdf": train_df,
            },
            constant_dict={
                "sdf_query_points": normalized_query_points,
                # "data_module": self,
            },
            path=self.data_dir,
            localtion_norm=location_norm_fn,
            # pressure_norm=copy.deepcopy(pressure_normalization).encode,
            pressure_norm=None,
            area_norm=area_norm_fn,
            indices=train_indices,
        )

        self._test_data = VariableDictDatasetTrackBWithConstant(
            data_dict={
                "sdf": test_df,
            },
            constant_dict={
                "sdf_query_points": normalized_query_points,
                # "data_module": self,
            },
            path=test_data_dir,
            localtion_norm=location_norm_fn,
            # pressure_norm=copy.deepcopy(pressure_normalization).encode,
            pressure_norm=None,
            area_norm=area_norm_fn,
            indices=test_indices,
        )

        if closest_points_to_query:
            self._train_data.data_dict["closest_points"] = train_closest_points
            self._test_data.data_dict["closest_points"] = test_closest_points

        self._aggregatable = list(self._train_data.data_dict.keys()) + list(
            self._train_data.constant_dict.keys()
        )

        self.output_normalization = pressure_normalization

    def get_mesh_path(self, data_dir: Path, subfolder: str, mesh_ind: int) -> Path:
        return data_dir / subfolder / ("mesh_" + str(mesh_ind).zfill(4) + ".ply")

    def get_pressure_data_path(
        self, data_dir: Path, subfolder: str, mesh_ind: int
    ) -> Path:
        return data_dir / subfolder / ("press_" + str(mesh_ind).zfill(4) + ".npy")

    def get_wss_data_path(self, data_dir: Path, subfolder: str, mesh_ind: int) -> Path:
        return (
            data_dir
            / subfolder
            / ("wallshearstress_" + str(mesh_ind).zfill(4) + ".npy")
        )

    def load_wss(self, data_dir: Path, subfolder: str, mesh_index: int) -> np.ndarray:
        wss_path = self.get_wss_data_path(data_dir, subfolder, mesh_index)
        assert wss_path.exists(), "wallshearstress data does not exist"
        wss = np.load(wss_path).astype(np.float32)
        return wss

    def load_pressure(
        self, data_dir: Path, subfolder: str, mesh_index: int
    ) -> np.ndarray:
        press_path = self.get_pressure_data_path(data_dir, subfolder, mesh_index)
        # print(press_path)
        assert press_path.exists(), "Pressure data does not exist"
        press = np.load(press_path).reshape((-1,)).astype(np.float32)
        return press

    def load_centroid(
        self, data_dir: Path, subfolder: str, mesh_index: int
    ) -> np.ndarray:
        centroid_path = (
            data_dir / subfolder / ("centroid_" + str(mesh_index).zfill(4) + ".npy")
        )
        assert centroid_path.exists(), "Centroid data does not exist"
        centroid = np.load(centroid_path).reshape((1, -1, 3)).astype(np.float32)
        return centroid

    def compute_df(
        self, mesh: Union[Path, o3d.t.geometry.TriangleMesh], query_points
    ) -> np.ndarray:
        if isinstance(mesh, Path):
            mesh = self.load_mesh(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        distance = scene.compute_distance(query_points).numpy()
        return distance

    def df_from_mesh(
        self, mesh_path: Path, query_points: np.ndarray, closest_points: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        mesh = self.load_mesh(mesh_path)
        df = self.compute_df(mesh, query_points)  # (64, 64, 64)

        if closest_points:
            closest_points = self.closest_points_to_query_from_mesh(mesh, query_points)
        else:
            closest_points = None
        return df, closest_points

    # def info_normalization(
    #     self, info: dict, min_bounds: List[float], max_bounds: List[float]
    # ) -> dict:
    #     """
    #     Normalize info to [0, 1].
    #     """
    #     for i, (k, v) in enumerate(info.items()):
    #         info[k] = (v - min_bounds[i]) / (max_bounds[i] - min_bounds[i])
    #     return info

    def area_normalization(
        self, area: torch.Tensor, min_bounds: float, max_bounds: float
    ) -> torch.Tensor:
        """
        Normalize info to [0, 1].
        """
        return (area - min_bounds) / (max_bounds - min_bounds)

    # def wss_normalization(
    #     self,
    #     area: torch.Tensor,
    #     min_bounds,
    #     max_bounds,
    # ) -> torch.Tensor:
    #     """
    #     Normalize info to [0, 1].
    #     """
    #     return (area - min_bounds) / (max_bounds - min_bounds)

    def collate_fn_paddle(self, batch):
        aggr_dict = {}
        for key in self._aggregatable:
            aggr_dict.update(
                {key: torch.stack(x=[data_dict[key] for data_dict in batch])}
            )
        remaining = list(
            set(batch[0].keys()) - set(self._aggregatable)
        )  # 计算第一个数据字典中所有键的集合与 self._aggregatable 的差集，得到剩余的键。
        for key in remaining:
            new_mini_batch_list = [data_dict[key] for data_dict in batch]
            if len(new_mini_batch_list) == 1:
                aggr_dict.update({key: new_mini_batch_list[0]})
            else:
                aggr_dict.update({key: new_mini_batch_list})

                # TODO for competitor : because centroid is not the same length, so a padding strategy may be needed.
                raise NotImplementedError(
                    "Not implemented for more than one element in the batch."
                )
        return aggr_dict

    def collate_fn(self, batch):
        aggr_dict = {}
        for key in self._aggregatable:
            aggr_dict.update(
                {key: torch.stack([data_dict[key] for data_dict in batch])}
            )  # 这里的值是把batchsize堆在一个tensor

        remaining = list(set(batch[0].keys()) - set(self._aggregatable))
        # ['pressure' ([104818]), 'info'len()8, 'areas'([104818]), 'centroids'([104818, 3]), 'vertices'([52692, 3])]
        for key in remaining:
            aggr_dict.update(
                {key: [data_dict[key] for data_dict in batch]}
            )  # 这里的值是一个列表，因为每个样本长度都不一样

        return aggr_dict


class AhmedBodyDataModule(CFDSDFDataModule):
    def __init__(
        self,
        data_dir,
        n_train: int = 500,
        n_test: int = 51,
        spatial_resolution: Tuple[int, int, int] = None,
        query_points=None,
        eps=1e-2,
        closest_points_to_query=True,
    ):
        BaseDataModule.__init__(self)

        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        data_dir = data_dir.expanduser()
        assert data_dir.exists(), "Path does not exist"
        assert data_dir.is_dir(), "Path is not a directory"
        self.data_dir = data_dir

        min_bounds, max_bounds = self.load_bound(
            data_dir, filename="global_bounds.txt", eps=eps
        )  # len()3

        min_info_bounds, max_info_bounds = self.load_bound(
            data_dir, filename="info_bounds.txt", eps=0.0
        )  # len()8
        # min_area_bound, max_area_bound = self.load_bound(
        #    data_dir, filename='area_bounds.txt', eps=0.0
        # )

        min_area_bound, max_area_bound = [0.0], [1.7468e-05]

        assert n_train <= 500, "Not enough training data"
        assert n_test <= 51, "Not enough testing data"
        if (n_train + n_test) < 551:
            warnings.warn(
                f"551 meshes are available, but {n_train + n_test} are requested."
            )

        train_indices = [j + 1 for j in range(n_train)]
        test_indices = [j + 1 for j in range(n_test)]
        train_mesh_pathes = [
            self.get_mesh_path(data_dir, "train", i) for i in train_indices
        ]
        test_mesh_pathes = [
            self.get_mesh_path(data_dir, "test", i) for i in test_indices
        ]

        if query_points is None:
            assert spatial_resolution is not None, "spatial_resolution must be given"
            tx = np.linspace(min_bounds[0], max_bounds[0], spatial_resolution[0])
            ty = np.linspace(min_bounds[1], max_bounds[1], spatial_resolution[1])
            tz = np.linspace(min_bounds[2], max_bounds[2], spatial_resolution[2])
            query_points = np.stack(
                np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            ).astype(np.float32)

        train_df_closest = [
            self.df_from_mesh(mesh_path, query_points, closest_points_to_query)
            for mesh_path in train_mesh_pathes
        ]
        train_df = torch.stack(
            [torch.Tensor(df) for df, _ in train_df_closest]
        )  # ([B, 64, 64, 64])
        if closest_points_to_query:
            train_closest_points = torch.stack(
                [torch.Tensor(closest) for _, closest in train_df_closest]
            )  # ([B, 64, 64, 64, 3])
        else:
            train_closest_points = None

        del train_df_closest

        test_df_closest = [
            self.df_from_mesh(mesh_path, query_points, closest_points_to_query)
            for mesh_path in test_mesh_pathes
        ]
        test_df = torch.stack([torch.Tensor(df) for df, _ in test_df_closest])
        if closest_points_to_query:
            test_closest_points = torch.stack(
                [torch.Tensor(closest) for _, closest in test_df_closest]
            )
        else:
            test_closest_points = None

        del test_df_closest

        # normalize pressure
        train_pressure = torch.cat(
            [
                torch.from_numpy(self.load_pressure(data_dir, "train", i))
                for i in train_indices
            ]
        )  # torch.Size([238177])=(104818,)和(133359,)竖向堆叠

        pressure_normalization = UnitGaussianNormalizer(
            train_pressure, eps=1e-6, reduce_dim=[0], verbose=False
        )

        min_bounds = torch.tensor(min_bounds)
        max_bounds = torch.tensor(max_bounds)

        normalized_query_points = self.location_normalization(
            torch.from_numpy(query_points), min_bounds, max_bounds
        ).permute(3, 0, 1, 2)

        if closest_points_to_query:
            train_closest_points = self.location_normalization(
                train_closest_points, min_bounds, max_bounds
            ).permute(0, 4, 1, 2, 3)

            test_closest_points = self.location_normalization(
                test_closest_points, min_bounds, max_bounds
            ).permute(0, 4, 1, 2, 3)

        location_norm_fn = lambda x: self.location_normalization(
            x, min_bounds, max_bounds
        )
        info_norm_fn = lambda x: self.info_normalization(
            x, min_info_bounds, max_info_bounds
        )
        area_norm_fn = lambda x: self.area_normalization(
            x, min_area_bound[0], max_area_bound[0]
        )
        self._train_data = VariableDictDatasetWithConstant(
            {
                "df": train_df,
            },
            {"df_query_points": normalized_query_points},
            self.data_dir / "train",
            localtion_norm=location_norm_fn,
            pressure_norm=copy.deepcopy(pressure_normalization).encode,
            info_norm=info_norm_fn,
            area_norm=area_norm_fn,
        )
        self._test_data = VariableDictDatasetWithConstant(
            {
                "df": test_df,
            },
            {"df_query_points": normalized_query_points},
            self.data_dir / "test",
            localtion_norm=location_norm_fn,
            pressure_norm=copy.deepcopy(pressure_normalization).encode,
            info_norm=info_norm_fn,
            area_norm=area_norm_fn,
        )

        if closest_points_to_query:
            self._train_data.data_dict["closest_points"] = train_closest_points
            self._test_data.data_dict["closest_points"] = test_closest_points

        self._aggregatable = list(self._train_data.data_dict.keys()) + list(
            self._train_data.constant_dict.keys()
        )

        self.output_normalization = pressure_normalization

    def get_mesh_path(self, data_dir: Path, subfolder: str, mesh_ind: int) -> Path:
        return data_dir / subfolder / ("mesh_" + str(mesh_ind).zfill(3) + ".ply")

    def get_pressure_data_path(
        self, data_dir: Path, subfolder: str, mesh_ind: int
    ) -> Path:
        return data_dir / subfolder / ("press_" + str(mesh_ind).zfill(3) + ".npy")

    def load_pressure(
        self, data_dir: Path, subfolder: str, mesh_index: int
    ) -> np.ndarray:
        press_path = self.get_pressure_data_path(data_dir, subfolder, mesh_index)
        assert press_path.exists(), "Pressure data does not exist"
        press = (
            np.load(press_path).reshape((-1,)).astype(np.float32)
        )  # (104818,) (133359,)
        return press

    def compute_df(
        self, mesh: Union[Path, o3d.t.geometry.TriangleMesh], query_points
    ) -> np.ndarray:
        if isinstance(mesh, Path):
            mesh = self.load_mesh(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        distance = scene.compute_distance(query_points).numpy()
        return distance

    def df_from_mesh(
        self,
        mesh_path: Path,
        query_points: np.ndarray,
        closest_points: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mesh = self.load_mesh(mesh_path)
        df = self.compute_df(mesh, query_points)
        if closest_points:
            closest_points = self.closest_points_to_query_from_mesh(mesh, query_points)
        else:
            closest_points = None

        return df, closest_points

    def info_normalization(
        self,
        info: dict,
        min_bounds: List[float],
        max_bounds: List[float],
    ) -> dict:
        """
        Normalize info to [0, 1].
        """
        for i, (k, v) in enumerate(info.items()):
            info[k] = (v - min_bounds[i]) / (max_bounds[i] - min_bounds[i])

        return info

    def area_normalization(
        self,
        area: torch.Tensor,
        min_bounds: float,
        max_bounds: float,
    ) -> torch.Tensor:
        """
        Normalize info to [0, 1].
        """
        return (area - min_bounds) / (max_bounds - min_bounds)

    def collate_fn(self, batch):
        aggr_dict = {}
        for key in self._aggregatable:
            aggr_dict.update(
                {key: torch.stack([data_dict[key] for data_dict in batch])}
            )  # 这里的值是把batchsize堆在一个tensor

        remaining = list(set(batch[0].keys()) - set(self._aggregatable))
        # ['pressure' ([104818]), 'info'len()8, 'areas'([104818]), 'centroids'([104818, 3]), 'vertices'([52692, 3])]
        for key in remaining:
            aggr_dict.update(
                {key: [data_dict[key] for data_dict in batch]}
            )  # 这里的值是一个列表，因为每个样本长度都不一样

        return aggr_dict


class CarDataModule:
    pass


class TestCFD(unittest.TestCase):
    def __init__(self, methodName: str, data_path: str) -> None:
        super().__init__(methodName)
        self.data_path = data_path

    def test_cfd(self):
        dm = CFDDataModule(
            self.data_path,
            n_train=10,
            n_test=10,
        )
        tl = dm.train_dataloader(batch_size=2, shuffle=True)
        for batch in tl:
            for k, v in batch.items():
                print(k, v.shape)
            break

    def test_cfd_grid(self):
        dm = CFDSDFDataModule(
            self.data_path,
            n_train=10,
            n_test=10,
            spatial_resolution=(64, 64, 64),
        )
        tl = dm.train_dataloader(batch_size=2, shuffle=True)
        for batch in tl:
            for k, v in batch.items():
                print(k, v.shape)
            break


class TestAhmed(unittest.TestCase):
    def __init__(self, methodName: str, data_path: str) -> None:
        super().__init__(methodName)
        self.data_path = data_path

    def test_ahmed(self):
        dm = AhmedBodyDataModule(
            self.data_path,
            n_train=10,
            n_test=10,
            spatial_resolution=(64, 64, 64),
        )

        tl = dm.train_dataloader(batch_size=2, shuffle=True)
        for batch in tl:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(k, v.shape)
                else:
                    print(k)
                    for j in range(len(v)):
                        if isinstance(v[j], dict):
                            print(v[j])
                        else:
                            print(v[j].shape)
            break


if __name__ == "__main__":
    data_dir_cfd = Path("~/datasets/geono/car-pressure-data").expanduser()
    data_dir_ahmed = Path("~/datasets/geono/ahmed").expanduser()

    # Unittest
    test_suite = unittest.TestSuite()
    # TestCFD and setup the path
    test_suite.addTest(TestCFD("test_cfd", data_dir_cfd))
    test_suite.addTest(TestCFD("test_cfd_grid", data_dir_cfd))
    test_suite.addTest(TestAhmed("test_ahmed", data_dir_ahmed))
    unittest.TextTestRunner().run(test_suite)
