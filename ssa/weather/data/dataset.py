from __future__ import annotations
from enum import Enum, auto
import logging
import os
from pathlib import Path
from typing import ClassVar, Union
from urllib.request import urlopen

from bolts.data.datasets.base import PBDataset
from kit import parsable
from kit.misc import str_to_enum
import pandas as pd
import requests
import torch
from tqdm import tqdm

__all__ = [
    "DataSplit",
    "ImputationMethod",
    "WeatherDataset",
]


def download_from_url(
    url: str,
    *,
    dst: str | Path,
    logger: logging.Logger | None = None,
) -> None:
    """Download from a url."""
    logger = logging.getLogger(__name__) if logger is None else logger

    if isinstance(dst, str):
        dst = Path(dst)

    if not dst.exists():
        logger.info(f"Downloading file {dst.name} from address '{url}'.")

        file_size = int(urlopen(url).info().get("Content-Length", -1))
        first_byte = os.path.getsize(dst) if dst.exists() else 0

        if first_byte < file_size:
            header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
            pbar = tqdm(
                total=file_size,
                initial=first_byte,
                unit="B",
                unit_scale=True,
                desc=url.split("/")[-1],
            )
            req = requests.get(url, headers=header, stream=True)
            with (open(dst, "ab")) as f:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(1024)
            pbar.close()
        else:
            logger.info(f"File '{dst.name}' already downloaded.")
    if dst.suffix == ".tar":
        if dst.with_suffix("").exists():
            logger.info(f"File '{dst.name}' already extracted.")
        else:
            import tarfile

            logger.info(
                f"Extracting '{dst.resolve()}' to {dst.parent.resolve()}; this could take a while."
            )
            with tarfile.TarFile(dst, "r") as fhandle:
                fhandle.extractall(dst.parent)


class DataSplit(Enum):
    train = auto()
    dev = auto()
    eval = auto()


class ImputationMethod(Enum):
    mean = auto()
    median = auto()
    none = auto()  # drop NaN-containing rows


class WeatherDataset(PBDataset):

    _URL: ClassVar[
        str
    ] = "https://storage.yandexcloud.net/yandex-research/shifts/weather/canonical-trn-dev-data.tar"
    _BASE_FOLDER: ClassVar[str] = "weather"
    _TARGET: ClassVar[str] = "fact_temperature"
    _TRAIN_COLS: ClassVar[list[str]] = [
        "fact_time",
        "fact_latitude",
        "fact_longitude",
        "fact_temperature",
        "fact_cwsm_class",
        "climate",
        "topography_bathymetry",
        "sun_elevation",
        "climate_temperature",
        "climate_pressure",
        "cmc_0_0_0_1000",
        "cmc_0_0_0_2_grad",
        "cmc_0_0_0_2_interpolated",
        "cmc_0_0_0_2_next",
        "cmc_0_0_0_2",
        "cmc_0_0_0_500",
        "cmc_0_0_0_700",
        "cmc_0_0_0_850",
        "cmc_0_0_0_925",
        "cmc_0_0_6_2",
        "cmc_0_0_7_1000",
        "cmc_0_0_7_2",
        "cmc_0_0_7_500",
        "cmc_0_0_7_700",
        "cmc_0_0_7_850",
        "cmc_0_0_7_925",
        "cmc_0_1_0_0",
        "cmc_0_1_11_0",
        "cmc_0_1_65_0",
        "cmc_0_1_66_0",
        "cmc_0_1_67_0",
        "cmc_0_1_68_0",
        "cmc_0_1_7_0",
        "cmc_0_2_2_10",
        "cmc_0_2_2_1000",
        "cmc_0_2_2_500",
        "cmc_0_2_2_700",
        "cmc_0_2_2_850",
        "cmc_0_2_2_925",
        "cmc_0_2_3_10",
        "cmc_0_2_3_1000",
        "cmc_0_2_3_500",
        "cmc_0_2_3_700",
        "cmc_0_2_3_850",
        "cmc_0_2_3_925",
        "cmc_0_3_0_0",
        "cmc_0_3_0_0_next",
        "cmc_0_3_1_0",
        "cmc_0_3_5_1000",
        "cmc_0_3_5_500",
        "cmc_0_3_5_700",
        "cmc_0_3_5_850",
        "cmc_0_3_5_925",
        "cmc_0_6_1_0",
        "cmc_available",
        "cmc_horizon_h",
        "cmc_precipitations",
        "cmc_timedelta_s",
        "gfs_2m_dewpoint",
        "gfs_a_vorticity",
        "gfs_available",
        "gfs_cloudness",
        "gfs_clouds_sea",
        "gfs_horizon_h",
        "gfs_humidity",
        "gfs_precipitable_water",
        "gfs_precipitations",
        "gfs_pressure",
        "gfs_r_velocity",
        "gfs_soil_temperature",
        "gfs_soil_temperature_available",
        "gfs_temperature_10000",
        "gfs_temperature_15000",
        "gfs_temperature_20000",
        "gfs_temperature_25000",
        "gfs_temperature_30000",
        "gfs_temperature_35000",
        "gfs_temperature_40000",
        "gfs_temperature_45000",
        "gfs_temperature_5000",
        "gfs_temperature_50000",
        "gfs_temperature_55000",
        "gfs_temperature_60000",
        "gfs_temperature_65000",
        "gfs_temperature_7000",
        "gfs_temperature_70000",
        "gfs_temperature_75000",
        "gfs_temperature_80000",
        "gfs_temperature_85000",
        "gfs_temperature_90000",
        "gfs_temperature_92500",
        "gfs_temperature_95000",
        "gfs_temperature_97500",
        "gfs_temperature_sea",
        "gfs_temperature_sea_grad",
        "gfs_temperature_sea_interpolated",
        "gfs_temperature_sea_next",
        "gfs_timedelta_s",
        "gfs_total_clouds_cover_high",
        "gfs_total_clouds_cover_low",
        "gfs_total_clouds_cover_middle",
        "gfs_u_wind",
        "gfs_v_wind",
        "gfs_wind_speed",
        "wrf_available",
        "wrf_t2",
        "wrf_t2_next",
        "wrf_psfc",
        "wrf_rh2",
        "wrf_wind_u",
        "wrf_wind_v",
        "wrf_rain",
        "wrf_snow",
        "wrf_graupel",
        "wrf_hail",
        "wrf_t2_interpolated",
        "wrf_t2_grad",
        "cmc_0_1_65_0_grad",
        "cmc_0_1_65_0_next",
        "cmc_0_1_66_0_grad",
        "cmc_0_1_66_0_next",
        "cmc_0_1_67_0_grad",
        "cmc_0_1_67_0_next",
        "cmc_0_1_68_0_grad",
        "cmc_0_1_68_0_next",
        "gfs_2m_dewpoint_grad",
        "gfs_2m_dewpoint_next",
        "gfs_total_clouds_cover_low_grad",
        "gfs_total_clouds_cover_low_next",
    ]

    @parsable
    def __init__(
        self,
        root: Union[Path, str],
        split: Union[DataSplit, str],
        download: bool = False,
        imputation_method: Union[ImputationMethod, str] = ImputationMethod.mean,
    ) -> None:
        if isinstance(root, str):
            root = Path(root)
        self._base_dir = root / self._BASE_FOLDER
        self._data_dir = self._base_dir / "data"
        if isinstance(imputation_method, str):
            imputation_method = str_to_enum(str_=imputation_method, enum=ImputationMethod)
        self.imputation_method = imputation_method
        self.download = download

        if self.download:
            if self._data_dir.exists():
                self.log("Files already downloaded and unzipped.")
            else:
                self._base_dir.mkdir(parents=True, exist_ok=True)
                download_from_url(
                    url=self._URL,
                    dst=self._base_dir / "canonical_trn_dev_data.tar",
                    logger=self.logger,
                )
                # Convert the csv files to tensors and save as .pt objects
                # - this is done as a pre-processing step to get around the
                # the huger start-up time introduced by conversion of the
                # very large arrays to tensors
                csv_files = tuple(self._data_dir.glob("**/*.csv"))
                for file in csv_files:
                    save_path = file.with_suffix(".pt")
                    if file.name == "train.csv":
                        # special instructions for loading this file because it's so large
                        df = pd.read_csv(
                            file,
                            dtype={
                                c: "float32" if c != "climate" else "object"
                                for c in self._TRAIN_COLS
                            },
                        )
                    else:
                        df = pd.read_csv(file)
                    # label-encode any categorical columns
                    cat_cols = df.select_dtypes("object")  # type: ignore
                    for col in cat_cols:
                        df[col] = df[col].factorize()[0]  # type: ignore
                    if self._TARGET in df.columns:
                        target = df.pop(self._TARGET)  # type: ignore
                        # Move the target to the end of the dataframe
                        df = pd.concat([df, target], axis=1)
                    data = torch.as_tensor(
                        df.astype("float32", copy=False).to_numpy(copy=False), dtype=torch.float32
                    )
                    torch.save(obj=data, f=save_path)

        elif not self._check_unzipped():
            raise RuntimeError(
                f"Data not found at location {self._data_dir.resolve()}. " "Have you downloaded it?"
            )

        if isinstance(split, str):
            str_to_enum(str_=split, enum=DataSplit)

        if split is DataSplit.train:
            data = torch.load(f=self._data_dir / "train.pt")
        else:
            dev_in = torch.load(f=self._data_dir / "dev_in.pt")
            dev_out = torch.load(f=self._data_dir / "dev_out.pt")
            data = torch.cat([dev_in, dev_out])

        if split is DataSplit.eval:
            x = data[:, 5:]
            y = None
        else:
            x = data[:, 5:-1]
            y = data[:, -1]

        # NaN-handling
        if y is not None:
            nan_mask_y = y.isnan()
            to_keep = nan_mask_y.view(nan_mask_y.size(0), -1).count_nonzero(dim=1) == 0
            x = x[to_keep]
            y = y[to_keep]

        nan_mask_x = x.isnan()
        if imputation_method is ImputationMethod.none:
            to_keep = nan_mask_x.count_nonzero(dim=1) == 0
            x = x[to_keep]
            if y is not None:
                y = y[to_keep]
        else:
            if imputation_method is ImputationMethod.mean:
                num_non_nan = (~nan_mask_x).count_nonzero()
                fill_values = torch.nansum(x, dim=0) / num_non_nan
            elif imputation_method is ImputationMethod.median:
                fill_values = torch.nanmedian(x, dim=0).values
            else:
                fill_values = torch.zeros(x.size(1))
            row_idxs, col_idxs = nan_mask_x.nonzero(as_tuple=True)
            x[row_idxs, col_idxs] = fill_values[col_idxs]

        super().__init__(x=x, y=y)

    def _check_unzipped(self) -> bool:
        return self._data_dir.exists()
