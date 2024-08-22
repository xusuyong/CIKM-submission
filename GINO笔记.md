# GINO笔记


500个训练 111个测试
## car_pressure_data文件夹

.
├── data
├── watertight_global_bounds.txt
└── watertight_meshes.txt
data文件夹里有798个.ply表示顶点，和798个.npy表示压力。
mesh.ply可不只有顶点信息，有
- TriangleMesh on CPU:0 [3586 vertices (Float32) and 7168 triangles (Int64)].
- Vertex Attributes: None.
- Triangle Attributes: None.
## car-cfd

文件夹里有param{8}9个子文件夹，每个文件夹里有100哈希值命名的子文件夹，每个文件夹里有
├── cd.txt
├── hexvelo_smpl.vtk
├── param1.txt
├── param2.txt
├── press.npy
├── quadpress_smpl.vtk
└── velo.npy

## ahmed文件夹

.
├── area_bounds.txt
├── global_bounds.txt
├── info_bounds.txt
├── test
└── train
train和test子文件夹，train文件夹里有500个info{i}.pt，500个mesh{i}.ply，和500个press{i}.npy

成功运行的：

```sh
# 2340  万参数 35.59 秒
python train.py --config my_configs/carpressure/SDFFNOGNO.yaml --data_path ../../xsy_datasets/GINO_dataset/car-pressure-data
# 28    万参数   7.98 秒
python train.py --config my_configs/carpressure/GNO.yaml --data_path ../../xsy_datasets/GINO_dataset/car-pressure-data
# 1632  万参数 54.58 秒 43.22 秒
python train.py --config my_configs/carpressure/UNet.yaml --data_path ../../xsy_datasets/GINO_dataset/car-pressure-data
# 19838 万参数    5395秒，100个估计要5天
python train.py --config my_configs/carpressure/GNOFNOGNO.yaml --data_path ../../xsy_datasets/GINO_dataset/car-pressure-data
# 386   万参数 33秒
python train.py --config my_configs/carpressure/Transolver.yaml --data_path ../../xsy_datasets/GINO_dataset/car-pressure-data
python trainDP.py --config my_configs/carpressure/Transolver.yaml --data_path ../../xsy_datasets/GINO_dataset/car-pressure-data
python trainDDP.py --config my_configs/carpressure/Transolver.yaml --data_path ../../xsy_datasets/GINO_dataset/car-pressure-data

python infer.py --config my_configs/carpressure/Transolver.yaml
```

```sh
python train.py --config my_configs/trackB/TransolverTrackB.yaml
python train.py --config my_configs/trackB/GNOFNOGNOTrankB.yaml
python trainDP.py --config my_configs/trackB/TransolverTrackB.yaml
python trainDDP.py --config my_configs/trackB/TransolverTrackB.yaml
```

```sh
#3.6亿参数
python train.py --config my_configs/ahmed/FNOGNOAhmed.yaml --data_path ../../xsy_datasets/GINO_dataset/ahmed 
#1.9亿参数
python train.py --config my_configs/ahmed/FNOInterpAhmed.yaml --data_path ../../xsy_datasets/GINO_dataset/ahmed
#1.9亿参数
python train.py --config my_configs/ahmed/GNOFNOGNOAhmed.yaml --data_path ../../xsy_datasets/GINO_dataset/ahmed
#1.9亿参数
python train.py --config my_configs/ahmed/GNOFNOGNOAhmedWeighted.yaml --data_path ../../xsy_datasets/GINO_dataset/ahmed
# 1633   万参数 ! RuntimeError: The size of tensor a (63945) must match the size of tensor b (127889) at non-singleton dimension 1
python train.py --config my_configs/ahmed/UNetAhmed.yaml --data_path ../../xsy_datasets/GINO_dataset/ahmed
```
## SDFFNOGNO

grid_sdf先lifting，又fno_blocks

SpectralConv是小框，fnoblocks是大框


- TFNO

ModuleList(
  (0-3): 4 x ComplexTuckerTensor(shape=(32, 32, 16, 9), rank=(26, 26, 13, 7))
)

- FNO

ModuleList(
  (0-3): 4 x ComplexDenseTensor(shape=torch.Size([32, 32, 16, 9]), rank=None)
)

主要就是SpectralConv里的R不同

## 原始数据集链接

http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip

B1_L8_Dim3_Slice64_l2loss