from __future__ import annotations
from dataclasses import astuple, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from kit import implements
from kit.torch import SequentialBatchSampler, TrainingMode
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch._six import string_classes
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data._utils.collate import (
    default_collate_err_msg_format,
    np_str_obj_array_pattern,
)

from ssa.weather.data.dataset import WeatherDataset, WeatherDatasetTest


def pb_default_collate(batch: list[Any]) -> Any:
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        ndims = elem.dim()
        if (ndims > 0) and ((ndims % 2) == 0):
            return torch.cat(batch, dim=0, out=out)
        else:
            return torch.stack(batch, dim=0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        elem = batch[0]
        if elem_type.__name__ == "ndarray":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return pb_default_collate([torch.as_tensor(b) for b in batch])
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, Mapping):
        return {key: pb_default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(pb_default_collate(samples) for samples in zip(*batch)))
    elif is_dataclass(elem):  # dataclass
        return elem_type(*pb_default_collate([astuple(sample) for sample in batch]))
    elif isinstance(elem, (tuple, list)):
        transposed = zip(*batch)
        return [pb_default_collate(samples) for samples in transposed]
    raise TypeError(default_collate_err_msg_format.format(elem_type))


class Weather(pl.LightningDataModule):
    _train_data: Dataset
    _val_test_data: Dataset

    def __init__(
        self,
        *,
        data_dir: Path,
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 47,
        persist_workers: bool = False,
        pin_memory: bool = True,
        training_mode: TrainingMode = TrainingMode.epoch,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.persist_workers = persist_workers
        self.pin_memory = pin_memory
        self.training_mode = training_mode

    def prepare_data(self) -> None:
        self._train_data = WeatherDataset(self.data_dir)
        self._val_test_data = WeatherDatasetTest(self.data_dir)

    def make_dataloader(
        self,
        ds: Dataset,
        *,
        shuffle: bool = False,
        drop_last: bool = False,
        batch_sampler: Sampler[Sequence[int]] | None = None,
    ) -> DataLoader:
        """Make DataLoader."""
        return DataLoader(
            ds,
            batch_size=1 if batch_sampler is not None else self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.persist_workers,
            batch_sampler=batch_sampler,
            collate_fn=pb_default_collate,
        )

    def train_dataloader(
        self, *, shuffle: bool = False, drop_last: bool = False, batch_size: int | None = None
    ) -> DataLoader:
        batch_size = self.batch_size if batch_size is None else batch_size

        batch_sampler = SequentialBatchSampler(
            data_source=self._train_data,  # type: ignore
            batch_size=batch_size,
            shuffle=shuffle,
            training_mode=self.training_mode,
            drop_last=drop_last,
        )
        return self.make_dataloader(ds=self._train_data, batch_sampler=batch_sampler)

    @implements(pl.LightningDataModule)
    def val_dataloader(self) -> DataLoader:
        return self.make_dataloader(ds=self._val_test_data)

    @implements(pl.LightningDataModule)
    def test_dataloader(self) -> DataLoader:
        return self.make_dataloader(ds=self._val_test_data)
