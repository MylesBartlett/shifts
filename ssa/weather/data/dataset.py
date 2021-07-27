from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class BinaryTuple(NamedTuple):
    x: Tensor
    y: Tensor


class WeatherDatasetTest(Dataset):
    def __init__(self, data_dir: Path):
        df_dev_in = pd.read_csv(data_dir / "weather" / "dev_in.csv")
        df_dev_out = pd.read_csv(data_dir / "weather" / "dev_out.csv")
        df_test = pd.concat([df_dev_in, df_dev_out])

        x = df_test.iloc[:, 6:]
        y = df_test['fact_temperature']
        self.x = torch.tensor(x.values)
        self.y = torch.tensor(y.values)

    def __getitem__(self, index) -> BinaryTuple:
        return BinaryTuple(x=self.x[index], y=self.y[index])


class WeatherDataset(Dataset):
    def __init__(self, data_dir: Path):
        df_train = pd.read_csv(data_dir / "weather" / "train.csv")

        x = df_train.iloc[:, 6:]
        y = df_train['fact_temperature']
        self.x = torch.tensor(x.values)
        self.y = torch.tensor(y.values)

    def __getitem__(self, index) -> T_co:
        return BinaryTuple(x=self.x[index], y=self.y[index])
