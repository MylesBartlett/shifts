from __future__ import annotations
from enum import Enum
from typing import Optional

from conduit.data import CdtDataModule
from conduit.data.structures import TrainValTestSplit
from ranzen import implements, parsable
from ranzen.torch import TrainingMode

from ssa import transforms as T
from ssa.weather.data.dataset import DataSplit, ImputationMethod, WeatherDataset

__all__ = ["WeatherDataModule"]


class NormalizationMethod(Enum):
    minmax = T.MinMaxNormalization
    quantile = T.QuantileNormalization
    zscore = T.ZScoreNormalization


class WeatherDataModule(CdtDataModule):
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
        imputation_method: ImputationMethod = ImputationMethod.mean,
        feature_normalizer: NormalizationMethod = NormalizationMethod.zscore,
        target_normalizer: NormalizationMethod = NormalizationMethod.zscore,
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
        self.feature_transform = feature_normalizer.value(inplace=True)
        self.target_transform = target_normalizer.value(inplace=True)
        self.imputation_method = imputation_method

    def prepare_data(self) -> None:
        WeatherDataset(root=self.root, split=DataSplit.train, download=True)

    @implements(CdtDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        data = {
            f"{split}": WeatherDataset(
                root=self.root, split=split, imputation_method=self.imputation_method
            )
            for split in DataSplit
        }
        # Feature normalization
        self.feature_transform.fit_transform(data[f"{DataSplit.dev}"].x)
        for split in (DataSplit.train, DataSplit.eval):
            self.feature_transform.transform(data[f"{split}"].x)
        # Target normalization
        self.target_transform.fit_transform(data[f"{DataSplit.dev}"].y)
        self.target_transform.transform(data[f"{DataSplit.train}"].y)

        return TrainValTestSplit(
            train=data[f"{DataSplit.train}"],
            val=data[f"{DataSplit.dev}"],
            test=data[f"{DataSplit.eval}"],
        )
