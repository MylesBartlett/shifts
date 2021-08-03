from __future__ import annotations
from typing import Optional

from bolts.data import PBDataModule
from bolts.data.structures import TrainValTestSplit
from kit import implements, parsable
from kit.torch import TrainingMode

from ssa.weather.data.dataset import DataSplit, WeatherDataset

__all__ = ["WeatherDataModule"]


class WeatherDataModule(PBDataModule):
    @parsable
    def __init__(
        self,
        *,
        root: str,
        train_batch_size: int = 100,
        eval_batch_size: Optional[int] = 256,
        num_workers: int = 0,
        seed: int = 47,
        persist_workers: bool = False,
        pin_memory: bool = True,
        training_mode: TrainingMode = TrainingMode.epoch,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            seed=seed,
            persist_workers=persist_workers,
            pin_memory=pin_memory,
            training_mode=training_mode,
        )
        self.root = root

    def prepare_data(self) -> None:
        WeatherDataset(root=self.root, split=DataSplit.train, download=True)

    @implements(PBDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        train_data = WeatherDataset(root=self.root, split=DataSplit.train)
        val_data = WeatherDataset(root=self.root, split=DataSplit.dev)
        # The (unlabeled) evaluation data is scheduled to be released in October
        # -- let's simply set the validation data as the test data for now
        test_data = WeatherDataset(root=self.root, split=DataSplit.eval)
        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
