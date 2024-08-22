# GeoNO

- Car CFD data: https://drive.google.com/file/d/1SWU-b4GrfFkWUrvxgEVe4XuU9iVvhrSg/view?usp=sharing
- Pressure data for SDF: https://drive.google.com/file/d/1Vb740MGw7dMN943bRTNvy_f_qFMB9sjX/view?usp=sharing
- Ahmed Body surface: https://drive.google.com/file/d/1Y7mscC4ohxy7tMfZhUJAynrzWX2_nt5a/view?usp=sharing

## Requirements

- Pytorch

```
pip install -r requirements.txt
```

## Training

```
# Example use cases
python train.py --config configs/GNO.yaml --logger_type tensorboard --data_path /datasets/car-pressure-data
python train.py --config configs/UNet.yaml --logger_type wandb
```
