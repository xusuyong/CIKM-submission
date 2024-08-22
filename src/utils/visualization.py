import io
import open3d as o3d
import numpy as np
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from PIL import Image


# Create a function to convert a figure to a NumPy array
def fig_to_numpy(fig: mpl.figure.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    im = Image.open(buf)
    im = np.array(im)
    buf.close()

    # Convert to valid image
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=-1)
    # if the image has 4 channels, remove the alpha channel
    if im.shape[-1] == 4:
        im = im[..., :3]
    # Convert to uint8 image
    if im.dtype != np.uint8:
        im = (im * 255).astype(np.uint8)
    return im


class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)[:, 0:3]


def vis_pressure(mesh_path, pressures, colormap="plasma", eps=0.5):
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    color_mapper = MplColorHelper(colormap, pressures.min(), pressures.max())
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        color_mapper.get_rgb(pressures[0, :])
    )

    meshes = [mesh]
    if pressures.shape[0] > 1:
        min_b = np.asarray(mesh.get_min_bound())
        max_b = np.asarray(mesh.get_max_bound())
        translation = np.array([max_b[0] - min_b[0] + eps, 0, 0])
        for j in range(1, pressures.shape[0]):
            new_mesh = o3d.io.read_triangle_mesh(mesh_path)
            new_mesh.translate(j * translation)
            new_mesh.vertex_colors = o3d.utility.Vector3dVector(
                color_mapper.get_rgb(pressures[j, :])
            )

            meshes.append(new_mesh)

    o3d.visualization.draw_geometries(meshes)


import re
import pyvista as pv


def extract_numbers(s):
    return [int(digit) for digit in re.findall(r"\d+", s)]


def write_to_vtk(out_dict, point_data_pos="press on mesh points", mesh_path=None):
    import meshio

    p = out_dict["pressure"]
    index = extract_numbers(mesh_path.name)[0]
    index = str(index).zfill(3)

    if point_data_pos == "press on mesh points":
        mesh = meshio.read(mesh_path)
        mesh.point_data["p"] = p
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

    print(f"write : ./output/{mesh_path.parent.name}_{index}.vtk")
    mesh.write(f"../../output/{mesh_path.parent.name}_{index}.vtk")


def create_visualization_subplots(mesh, pressure_name="p", n_points=100000):
    """
    Create subplots for visualizing the solid mesh, mesh with pressure, and point cloud with pressure.

    Parameters:
    mesh (pyvista.PolyData): The mesh to visualize.
    pressure_name (str): The name of the pressure field in the mesh's point data.
    n_points (int): Number of points to sample for the point cloud.
    """
    camera_position = [
        (-11.073024242161921, -5.621499358347753, 5.862225824910342),
        (1.458462064391673, 0.002314306982062475, 0.6792134746589196),
        (0.34000174095454166, 0.10379556639001211, 0.9346792479485448),
    ]
    # Sample points from the mesh for the point cloud
    if mesh.n_points > n_points:
        indices = np.random.choice(mesh.n_points, n_points, replace=False)
    else:
        indices = np.arange(mesh.n_points)


    # Exclude indices from 16 to 112
    exclude_indices = np.arange(16, 112)
    # indices = np.setdiff1d(indices, exclude_indices)
    sampled_points = mesh.points[indices]
    sampled_pressures = mesh.point_data[pressure_name][indices]

    # Create a point cloud with pressure data
    point_cloud = pv.PolyData(sampled_points)
    point_cloud[pressure_name] = sampled_pressures

    # Set up the plotter
    plotter = pv.Plotter(shape=(1, 3))

    # Solid mesh visualization
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh, color="lightgrey")
    plotter.add_text("Solid Mesh", position="upper_left")
    plotter.camera_position = camera_position

    # Mesh with pressure visualization
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh, scalars=pressure_name, cmap="jet")
    plotter.add_scalar_bar(title=pressure_name, vertical=True)
    plotter.add_text("Mesh with Pressure", position="upper_left")
    plotter.camera_position = camera_position

    # Point cloud with pressure visualization
    plotter.subplot(0, 2)
    plotter.add_points(
        point_cloud, scalars=pressure_name, cmap="jet", point_size=5,
        # clim=(-600, 400)#这个是原来的范围
    )
    plotter.add_scalar_bar(title=pressure_name, vertical=True)
    plotter.add_text("Point Cloud with Pressure", position="upper_left")
    plotter.camera_position = camera_position
    # Show the plot
    plotter.show()


if __name__ == "__main__":
    # from pathlib import Path

    # press_path = "/home/xusuyong/pythoncode/xsy_datasets/GINO_dataset/car-pressure-data/data/press_001.npy"
    # # press = np.load(press_path)[:, None].T
    # press = np.load(press_path).reshape((-1,)).astype(np.float32)  # (3682,)
    # press = np.concatenate((press[0:16], press[112:]), axis=0)[:, None]  # (3586,)
    # # vis_pressure(
    # #     "/home/xusuyong/pythoncode/xsy_datasets/GINO_dataset/car-pressure-data/data/mesh_001.ply",
    # #     press,
    # # )
    # press = {"pressure": press}
    # write_to_vtk(
    #     press,
    #     mesh_path=Path(
    #         "/home/xusuyong/pythoncode/xsy_datasets/GINO_dataset/car-pressure-data/data/mesh_001.ply"
    #     ),
    # )
    
    #下面是DirAverCar里面的
    # Load your mesh data here, ensure it has the pressure data in point_data
    # mesh = pv.read(
    #     "/home/xusuyong/pythoncode/myproj/GINO-submission/output/visualize/data_test_A_715.vtk"
    # )
    # create_visualization_subplots(mesh, pressure_name="p", n_points=100000)
    
    # mesh = pv.read(
    #     "/home/xusuyong/pythoncode/xsy_datasets/3D_GeoCA_dataset/car-cfd/training_data/param0/1a0bc9ab92c915167ae33d942430658c/hexvelo_smpl.vtk"
    # )
    # create_visualization_subplots(mesh, pressure_name="point_vectors", n_points=100000)
    
    mesh = pv.read(
        "/home/xusuyong/pythoncode/xsy_datasets/3D_GeoCA_dataset/car-cfd/training_data/param0/1a0bc9ab92c915167ae33d942430658c/quadpress_smpl.vtk"
    )
    create_visualization_subplots(mesh, pressure_name="point_scalars", n_points=100000)
