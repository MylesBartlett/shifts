from __future__ import annotations
from pathlib import Path

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
        root: Path,
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 47,
        persist_workers: bool = False,
        pin_memory: bool = True,
        training_mode: TrainingMode = TrainingMode.epoch,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            persist_workers=persist_workers,
            pin_memory=pin_memory,
            training_mode=training_mode,
        )
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.persist_workers = persist_workers
        self.pin_memory = pin_memory
        self.training_mode = training_mode

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
