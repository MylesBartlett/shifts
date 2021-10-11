from __future__ import annotations
from enum import Enum, auto
import logging
import os
from pathlib import Path
from typing import ClassVar, Union
from urllib.request import urlopen

from conduit.data.datasets.base import CdtDataset
from kit import parsable
from kit.misc import str_to_enum
import numpy as np
import polars as pls
from polars import datatypes as pldt
from polars.eager.frame import DataFrame
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


class WeatherDataset(CdtDataset):

    _URL: ClassVar[
        str
    ] = "https://storage.yandexcloud.net/yandex-research/shifts/weather/canonical-trn-dev-data.tar"
    _BASE_FOLDER: ClassVar[str] = "weather"
    _TARGET: ClassVar[str] = "fact_temperature"

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

        elif not self._check_unzipped():
            raise RuntimeError(
                f"Data not found at location {self._data_dir.resolve()}. " "Have you downloaded it?"
            )

        if isinstance(split, str):
            str_to_enum(str_=split, enum=DataSplit)

        def _load_data_from_csv(_filepath: Path) -> DataFrame:
            # first eead only ten entries and infer types from that
            df_10 = pls.read_csv(_filepath, stop_after_n_rows=100)
            # convert all numeric columns to float32
            dtypes = {}
            for col, dtype in zip(df_10.columns, df_10.dtypes):
                if dtype in (pls.datatypes.Float64, pls.datatypes.Int64, pls.datatypes.Int32):
                    dtype = pls.datatypes.Float32
                print(col, dtype)

                dtypes[col] = dtype
            del df_10  # try to free memory; not sure this does anything

            # now load the whole file
            df = pls.read_csv(_filepath, dtype=dtypes, low_memory=False)  # type: ignore
            # label-encode 'climate'
            df["climate"] = df["climate"].cast(pls.datatypes.Categorical).cast(pls.datatypes.UInt8)
            return df

        if split is DataSplit.train:
            data = _load_data_from_csv(_filepath=self._data_dir / "train.csv")
        else:
            dev_in = _load_data_from_csv(_filepath=self._data_dir / "dev_in.csv")
            dev_out = _load_data_from_csv(_filepath=self._data_dir / "dev_out.csv")
            data = pls.concat([dev_in, dev_out])

        if split is DataSplit.eval:
            y = None
        else:
            y = torch.from_numpy(data[self._TARGET].to_numpy())
            data.drop_in_place(self._TARGET)
        x = torch.from_numpy(data[:, 5:].to_numpy())

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
