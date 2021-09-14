from typing import Type

import pytest
import pytorch_lightning as pl

from ssa.weather.data import WeatherDataModule


@pytest.mark.parametrize("dm", [WeatherDataModule])
def test_dm(dm: Type[pl.LightningDataModule]):
    datamodule = dm(root='~/Data')
    datamodule.prepare_data()
    datamodule.setup()

    for batch in datamodule.train_dataloader():
        assert batch.x.isnan().sum() == 0
        assert batch.y.isnan().sum() == 0
