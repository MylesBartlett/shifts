#!/usr/bin/env python3
from __future__ import annotations

from bolts.data import BinarySample, PBDataModule
from bolts.structures import Stage
from kit import implements
from kit.torch import TrainingMode
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
import torch.distributions as td

from ..base_model import BaseVariationalModel, ShiftsBaseModel, ValStepOut
from ..due import ActivationFn, FCResNet

__all__ = ["SimpleRegression"]


class SimpleRegression(BaseVariationalModel):
    feature_extractor: nn.Module
    mean_std_net: nn.Module

    def __init__(
        self,
        activation_fn: ActivationFn = ActivationFn.relu,
        depth: int = 8,
        dropout_rate: float = 0.95,
        lr: float = 3.0e-4,
        lr_initial_restart: int = 10,
        lr_restart_mult: int = 2,
        lr_sched_interval: TrainingMode = TrainingMode.epoch,
        lr_sched_freq: int = 1,
        n_power_iterations: int = 1,
        num_features: int = 128,
        snorm_coeff: float = 0.95,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(
            lr=lr,
            lr_initial_restart=lr_initial_restart,
            lr_restart_mult=lr_restart_mult,
            lr_sched_interval=lr_sched_interval,
            lr_sched_freq=lr_sched_freq,
            weight_decay=weight_decay,
        )
        self.activation_fn = activation_fn
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.n_power_iterations = n_power_iterations
        self.num_features = num_features
        self.snorm_coeff = snorm_coeff
        # TODO: set the below with args?
        self._dist = td.Normal

    @implements(BaseVariationalModel)
    def _get_loss(self, variational_dist: td.Distribution, batch: BinarySample) -> Tensor:
        return -variational_dist.log_prob(batch.y).mean()

    @implements(BaseVariationalModel)
    def _inference_step(
        self, x: Tensor, batch_idx: int, dataloader_idx: int | None = None
    ) -> Tensor:
        variational_dist = self.forward(x)
        return torch.stack([variational_dist.mean, variational_dist.stddev], dim=-1)

    @implements(BaseVariationalModel)
    def forward(self, x: Tensor) -> td.Distribution:
        z = self.feature_extractor(x)
        mean_std: Tensor = self.mean_std_net(z)
        return self._dist(
            mean_std[:, 0].sigmoid(), mean_std[:, 1].sigmoid() + torch.finfo(torch.float32).eps
        )

    @implements(BaseVariationalModel)
    def validation_step(self, batch: BinarySample, batch_idx: int) -> ValStepOut:
        variational_dist = self.forward(batch.x)
        loss = self._get_loss(variational_dist, batch)
        self.val_mse(variational_dist.mean, batch.y)
        self.val_mae(variational_dist.mean, batch.y)
        self.log_dict(
            {
                f"{Stage.validate.value}/loss": float(loss.item()),  # type: ignore
                f"{Stage.validate.value}/mse": self.val_mse,
                f"{Stage.validate.value}/mae": self.val_mae,
            }
        )
        return ValStepOut(
            targets=batch.y.view(-1),
            pred_means=variational_dist.mean,
            pred_stddevs=variational_dist.stddev,
        )

    @implements(ShiftsBaseModel)
    def build(self, datamodule: PBDataModule, trainer: pl.Trainer) -> None:
        self.feature_extractor = FCResNet(
            in_channels=datamodule.dim_x[0],
            num_features=self.num_features,
            depth=self.depth,
            snorm_coeff=self.snorm_coeff,
            n_power_iterations=self.n_power_iterations,
            dropout_rate=self.dropout_rate,
            activation_fn=self.activation_fn,
        )
        self.mean_std_net = nn.Linear(self.num_features, 2)
