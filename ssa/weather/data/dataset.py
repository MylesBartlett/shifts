from __future__ import annotations
from enum import Enum, auto
from pathlib import Path
from typing import ClassVar, Union

from conduit.data.datasets.base import CdtDataset
from conduit.data.datasets.utils import UrlFileInfo, download_from_url
from kit import parsable
from kit.misc import str_to_enum
import polars as pl
from polars import datatypes as pldt
from polars.eager.frame import DataFrame
import torch

__all__ = [
    "DataSplit",
    "ImputationMethod",
    "WeatherDataset",
]


class DataSplit(Enum):
    train = auto()
    dev = auto()
    eval = auto()


class ImputationMethod(Enum):
    mean = auto()
    median = auto()
    none = auto()  # drop NaN-containing rows


class WeatherDataset(CdtDataset):

    _FILE_INFO: ClassVar[UrlFileInfo] = UrlFileInfo(
        name="canonical_trn_dev_data.tar",
        url="https://storage.yandexcloud.net/yandex-research/shifts/weather/canonical-trn-dev-data.tar",
    )
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
        self._data_dir = self._base_dir / "canonical_trn_dev_data" / "data"
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
                    file_info=self._FILE_INFO,
                    root=self._base_dir,
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
            df_10 = pl.read_csv(_filepath, stop_after_n_rows=2, infer_schema_length=2)
            # convert all numeric columns to float32
            dtypes = {}
            for col, dtype in zip(df_10.columns, df_10.dtypes):
                if dtype in (pldt.Float64, pldt.Int64, pldt.Int32):
                    dtype = pldt.Float32
                elif col == "climate":
                    dtype = pldt.Categorical

                dtypes[col] = dtype
            del df_10  # try to free memory; not sure this does anything

            # now load the whole file
            df = pl.read_csv(_filepath, dtype=dtypes, low_memory=False)  # type: ignore
            # label-encode 'climate'
            df["climate"] = df["climate"].cast(pldt.UInt8)
            return df

        if split is DataSplit.train:
            data = _load_data_from_csv(_filepath=self._data_dir / "train.csv")
        else:
            dev_in = _load_data_from_csv(_filepath=self._data_dir / "dev_in.csv")
            dev_out = _load_data_from_csv(_filepath=self._data_dir / "dev_out.csv")
            data = pl.concat([dev_in, dev_out])

        if split is DataSplit.eval:
            y = None
        else:
            y = torch.tensor(data[self._TARGET].to_numpy())
            data.drop_in_place(self._TARGET)  # type: ignore
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
