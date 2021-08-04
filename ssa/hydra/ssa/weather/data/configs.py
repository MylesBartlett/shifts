# Generated by configen, do not edit.
# See https://github.com/facebookresearch/hydra/tree/master/tools/configen
# fmt: off
# isort:skip_file
# flake8: noqa

from dataclasses import dataclass, field
from kit.torch.data import TrainingMode
from omegaconf import MISSING
from ssa.weather.data.dataset import ImputationMethod
from typing import Optional


@dataclass
class WeatherDataModuleConf:
    _target_: str = "ssa.weather.data.WeatherDataModule"
    root: str = MISSING
    train_batch_size: int = 100
    eval_batch_size: Optional[int] = 256
    num_workers: int = 0
    seed: int = 47
    persist_workers: bool = False
    pin_memory: bool = True
    training_mode: TrainingMode = TrainingMode.epoch
    imputation_method: ImputationMethod = ImputationMethod.mean