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

__all__ = ["DataSplit", "WeatherDataset"]


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


class WeatherDataset(PBDataset):

    _URL: ClassVar[
        str
    ] = "https://storage.yandexcloud.net/yandex-research/shifts/weather/canonical-trn-dev-data.tar"
    _BASE_FOLDER: ClassVar[str] = "weather"

    @parsable
    def __init__(
        self, root: Union[Path, str], split: Union[DataSplit, str], download: bool = False
    ) -> None:
        if isinstance(root, str):
            root = Path(root)
        self._base_dir = root / self._BASE_FOLDER
        self._data_dir = self._base_dir / "data"
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
        # usecols = list(range(6)) + ["fact_temperature"]
        usecols = [3] + list(range(6, 128))
        if split is DataSplit.train:
            df = pd.read_csv(self._data_dir / "train.csv", usecols=usecols)
        else:
            df_dev_in = pd.read_csv(self._data_dir / "dev_in.csv", usecols=usecols)
            df_dev_out = pd.read_csv(self._data_dir / "dev_out.csv", usecols=usecols)
            df = pd.concat([df_dev_in, df_dev_out])

        x = torch.as_tensor(df.iloc[:, 6:].to_numpy(), dtype=torch.float32)
        y = None if split is DataSplit.eval else torch.as_tensor(df['fact_temperature'].to_numpy())

        super().__init__(x=x, y=y)

    def _check_unzipped(self) -> bool:
        return self._data_dir.exists()
