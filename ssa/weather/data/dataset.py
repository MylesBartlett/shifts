from __future__ import annotations
from enum import Enum, auto
from pathlib import Path
from typing import ClassVar, List, Union

from conduit.data.datasets.tabular import CdtTabularDataset
from conduit.data.datasets.utils import UrlFileInfo, download_from_url
import polars as pl
from polars import datatypes as pldt
from polars.eager.frame import DataFrame
import torch

__all__ = [
    "DataSplit",
    "ImputationMethod",
    "WeatherDataset",
]

from ranzen import parsable, str_to_enum
from torch import Tensor


class DataSplit(Enum):
    train = auto()
    dev = auto()
    eval = auto()


class ImputationMethod(Enum):
    mean = auto()
    median = auto()
    none = auto()  # drop NaN-containing rows


class WeatherDataset(CdtTabularDataset):

    _FILE_INFO: ClassVar[List[UrlFileInfo]] = [
        UrlFileInfo(
            name="canonical_trn_dev_data.tar",
            url="https://storage.yandexcloud.net/yandex-research/shifts/weather/canonical-trn-dev-data.tar",
        ),
        UrlFileInfo(
            name="canonical_eval_data.tar",
            url="https://storage.yandexcloud.net/yandex-research/shifts/weather/canonical-eval-data.tar",
        ),
    ]
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
        self._trn_dev_data_dir = self._base_dir / "canonical_trn_dev_data" / "data"
        self._eval_data_dir = self._base_dir / "canonical_eval_data" / "canonical-eval-data"
        if isinstance(imputation_method, str):
            imputation_method = str_to_enum(str_=imputation_method, enum=ImputationMethod)
        self.imputation_method = imputation_method
        self.download = download

        if self.download:
            if self._trn_dev_data_dir.exists():
                self.log("Files already downloaded and unzipped.")
            else:
                self._base_dir.mkdir(parents=True, exist_ok=True)
                for file in self._FILE_INFO:
                    download_from_url(
                        file_info=file,
                        root=self._base_dir,
                        logger=self.logger,
                    )

        elif not self._check_unzipped():
            raise RuntimeError(
                f"Data not found at location {self._base_dir.resolve()}. " "Have you downloaded it?"
            )

        if isinstance(split, str):
            split = str_to_enum(str_=split, enum=DataSplit)

        x, y = self._load_x_y_pair(split)

        x, y = self._do_imputation(data=x, labels=y, imputation=imputation_method)

        super().__init__(x=x, y=y)

    def _load_split(self, split: DataSplit) -> DataFrame:

        if split is DataSplit.train:
            return self._load_data(filepath=self._trn_dev_data_dir / "train.csv")
        elif split is DataSplit.dev:
            dev_in = self._load_data(filepath=self._trn_dev_data_dir / "dev_in.csv")
            dev_out = self._load_data(filepath=self._trn_dev_data_dir / "dev_out.csv")
            return pl.concat([dev_in, dev_out])
        else:
            return self._load_data(filepath=self._eval_data_dir / "eval.csv")

    def _do_imputation(
        self, data: Tensor, *, labels: Tensor, imputation: ImputationMethod
    ) -> tuple[Tensor, Tensor]:
        nan_mask_x = data.isnan()
        if imputation is ImputationMethod.none:
            to_keep = nan_mask_x.count_nonzero(dim=1) == 0
            data = data[to_keep]
            if labels is not None:
                labels = labels[to_keep]
        else:
            if imputation is ImputationMethod.mean:
                num_non_nan = (~nan_mask_x).count_nonzero()
                fill_values = torch.nansum(data, dim=0) / num_non_nan
            elif imputation is ImputationMethod.median:
                fill_values = torch.nanmedian(data, dim=0).values
            else:
                fill_values = torch.zeros(data.size(1))
            row_idxs, col_idxs = nan_mask_x.nonzero(as_tuple=True)
            data[row_idxs, col_idxs] = fill_values[col_idxs]
        return data, labels

    def _load_x_y_pair(self, split: DataSplit) -> tuple[Tensor, Tensor]:
        data = self._load_split(split)
        if split is DataSplit.eval:
            y = None
        else:
            y = torch.tensor(data[self._TARGET].to_numpy())
            data.drop_in_place(self._TARGET)  # type: ignore
            data = data[:, 5:]
        x = torch.from_numpy(data.to_numpy())

        # NaN-handling
        if y is not None:
            nan_mask_y = y.isnan()
            to_keep = nan_mask_y.view(nan_mask_y.size(0), -1).count_nonzero(dim=1) == 0
            x = x[to_keep]
            y = y[to_keep]

        return x, y

    def _load_data(self, filepath: Path) -> DataFrame:
        # first eead only ten entries and infer types from that
        df_10 = pl.read_csv(filepath, stop_after_n_rows=2, infer_schema_length=2)
        # convert all numeric columns to float32
        dtypes = {}
        for col, dtype in zip(df_10.columns, df_10.dtypes):
            if dtype in (pldt.Float64, pldt.Int64, pldt.Int32):
                dtype = pldt.Float32
            elif col == "climate":
                dtype = pldt.Categorical

            dtypes[col] = dtype
        del df_10  # try to free memory; not sure this does anything

        data = pl.read_csv(filepath, dtype=dtypes, low_memory=False)  # type: ignore
        if "climate" in data.columns:
            data["climate"] = data["climate"].cast(pldt.UInt8)
        return data

    def _check_unzipped(self) -> bool:
        return self._trn_dev_data_dir.exists()
