from __future__ import annotations
from enum import Enum
from typing import Optional

from conduit.data import CdtDataModule
from conduit.data.structures import TrainValTestSplit
from kit import implements, parsable
from kit.torch import TrainingMode

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
        train_data = WeatherDataset(
            root=self.root, split=DataSplit.train, imputation_method=self.imputation_method
        )
        val_data = WeatherDataset(
            root=self.root, split=DataSplit.dev, imputation_method=self.imputation_method
        )
        # The (unlabeled) evaluation data is scheduled to be released in October
        # -- let's simply set the validation data as the test data for now
        # test_data = WeatherDataset(root=self.root, split=DataSplit.eval)
        # Feature normalization
        self.feature_transform.fit_transform(val_data.x)
        self.feature_transform.transform(train_data.x)
        # self.feature_transform.transform(test_data.x)
        # Target normalization
        self.target_transform.fit_transform(train_data.y)
        self.target_transform.transform(val_data.y)
        # self.feature_transform.transform(test_data.y)
        test_data = val_data

        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
