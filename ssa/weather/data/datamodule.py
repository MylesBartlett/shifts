from __future__ import annotations
from abc import abstractmethod
from typing import Optional

from bolts.data import PBDataModule
from bolts.data.structures import TrainValTestSplit
from kit import implements, parsable
from kit.torch import TrainingMode
import torch
from torch import Tensor

from ssa.weather.data.dataset import DataSplit, ImputationMethod, WeatherDataset

__all__ = ["WeatherDataModule"]


class TabularTransform:
    def __init__(self, inplace: bool = False) -> None:
        self.inplace = inplace

    @abstractmethod
    def fit(self, input: Tensor) -> TabularTransform:
        ...

    def fit_transform(self, input: Tensor) -> Tensor:
        self.fit(input)
        return self.transform(input)

    @abstractmethod
    def inverse_transform(self, output: Tensor) -> Tensor:
        ...

    @abstractmethod
    def transform(self, input: Tensor) -> Tensor:
        ...

    def __call__(self, input: Tensor) -> Tensor:
        return self.transform(input)


class ZScoreNormalization(TabularTransform):

    mean: Tensor
    std: Tensor

    @implements(TabularTransform)
    def fit(self, input: Tensor) -> ZScoreNormalization:
        self.std, self.mean = torch.std_mean(input, dim=0, keepdim=True, unbiased=True)
        return self

    @implements(TabularTransform)
    def inverse_transform(self, output: Tensor) -> Tensor:
        if self.inplace:
            output *= self.std
            output += self.mean
        else:
            output = (output * self.std) + self.mean
        return output

    @implements(TabularTransform)
    def transform(self, input: Tensor) -> Tensor:
        if self.inplace:
            input -= self.mean
            input /= self.std
        else:
            input = (input - self.mean) / self.std
        return input


class MinMaxNormalization(TabularTransform):

    orig_max: Tensor
    orig_min: Tensor
    orig_range: Tensor

    def __init__(self, new_min: float = 0.0, new_max: float = 1.0, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)
        if new_min > new_max:
            raise ValueError("'new_min' cannot be greater than 'new_max'.")
        self.new_min = new_min
        self.new_max = new_max
        self.new_range = self.new_max - self.new_min

    @implements(TabularTransform)
    def fit(self, input: Tensor) -> MinMaxNormalization:
        self.orig_min = torch.min(input, dim=0, keepdim=True).values
        self.orig_max = torch.max(input, dim=0, keepdim=True).values
        self.orig_range = self.orig_max - self.orig_min
        return self

    @implements(TabularTransform)
    def inverse_transform(self, output: Tensor) -> Tensor:
        output_std = (output - self.new_min) / (self.new_range)
        if self.inplace:
            output *= self.orig_range
            output += self.orig_min
        else:
            output = output_std * self.orig_range + self.orig_min
        return output

    @implements(TabularTransform)
    def transform(self, input: Tensor) -> Tensor:
        input_std = (input - self.orig_min) / (self.orig_range)
        if self.inplace:
            input *= self.new_range
            input += self.new_min
        else:
            input = input_std * self.new_range + self.new_min
        return input


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
        imputation_method: ImputationMethod = ImputationMethod.mean,
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
        self.feature_transform = MinMaxNormalization(inplace=True)
        self.target_transform = MinMaxNormalization(inplace=True)
        self.imputation_method = imputation_method

    def prepare_data(self) -> None:
        WeatherDataset(root=self.root, split=DataSplit.train, download=True)

    @implements(PBDataModule)
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
        self.feature_transform.fit_transform(train_data.x)
        self.feature_transform.transform(val_data.x)
        # self.feature_transform.transform(test_data.x)
        # Target normalization
        self.feature_transform.fit_transform(train_data.y)
        self.feature_transform.transform(val_data.y)
        # self.feature_transform.transform(test_data.y)
        test_data = val_data

        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
