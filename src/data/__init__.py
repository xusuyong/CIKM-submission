from .cfd_datamodule import CFDSDFDataModule, CFDDataModule, AhmedBodyDataModule, CFDNormalDataModule, CarDataModule, TrackBDataModule
from .velocity_datamodule import VelocityDataModule
from .cd_datamodule import CdDataModule

def instantiate_datamodule(config):
    if config.data_module == "CFDDataModule":
        return CFDDataModule(
            config.data_path,
            n_train=config.n_train,
            n_test=config.n_test,
        )
    elif config.data_module == "CFDNormalDataModule":
        return CFDNormalDataModule(
            config.data_path,
            n_train=config.n_train,
            n_test=config.n_test,
            use_multifi = config.use_multifi
        )
    elif config.data_module == "CarDataModule":
        assert config.sdf_spatial_resolution is not None
        return CarDataModule(
            config.data_dir,
            config.test_data_dir,
            config.n_train,
            config.n_test,
            spatial_resolution=config.sdf_spatial_resolution,
        )
    elif config.data_module == "TrackBDataModule":
        assert config.sdf_spatial_resolution is not None
        return TrackBDataModule(
            config.data_dir,
            config.test_data_dir,
            config.n_train,
            config.n_test,
            spatial_resolution=config.sdf_spatial_resolution,
            closest_points_to_query=config.closest_points_to_query,
            Require_sdf=config.Require_sdf
        )
    elif config.data_module == "CFDSDFDataModule":
        assert config.sdf_spatial_resolution is not None
        return CFDSDFDataModule(
            config.data_path,
            n_train=config.n_train,
            n_test=config.n_test,
            spatial_resolution=config.sdf_spatial_resolution,
        )
    elif config.data_module == "AhmedBodyDataModule":
        assert config.sdf_spatial_resolution is not None
        return AhmedBodyDataModule(
            config.data_path,
            n_train=config.n_train,
            n_test=config.n_test,
            spatial_resolution=config.sdf_spatial_resolution,
        )
    elif config.data_module == "VelocityDataModule":
        assert config.sdf_spatial_resolution is not None
        return VelocityDataModule(
            config.train_data_dir,
            config.test_data_dir,
            config.train_index_list,
            config.test_index_list,
        )
    elif config.data_module == "CdDataModule":
        assert config.sdf_spatial_resolution is not None
        return CdDataModule(
            config.train_data_dir,
            config.test_data_dir,
            config.train_index_list,
            config.test_index_list,
            config.num_points,
        )
    else:
        raise NotImplementedError(f"Unknown datamodule: {config.data_module}")
