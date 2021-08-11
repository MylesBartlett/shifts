#!/usr/bin/env python3
from __future__ import annotations
from typing import Mapping, NamedTuple

from bolts.data import BinarySample, NamedSample, PBDataModule
from bolts.structures import LRScheduler, Stage
from kit import implements
from kit.torch import TrainingMode
import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim
import torch.distributions as td
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from ssa.weather.assessment import f_beta_metrics

from ..due import ActivationFn, FCResNet

__all__ = ["SimpleRegression"]


class ValStepOut(NamedTuple):
    pred_means: Tensor
    pred_stddevs: Tensor
    targets: Tensor


class SimpleRegression(pl.LightningModule):
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
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_initial_restart = lr_initial_restart
        self.lr_restart_mult = lr_restart_mult
        self.lr_sched_interval = lr_sched_interval
        self.lr_sched_freq = lr_sched_freq
        self.activation_fn = activation_fn
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.n_power_iterations = n_power_iterations
        self.num_features = num_features
        self.snorm_coeff = snorm_coeff
        # TODO: set the below with args?
        self._dist = td.Normal

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()

    def _get_loss(self, variational_dist: td.Distribution, batch: BinarySample) -> Tensor:
        return -variational_dist.log_prob(batch.y).mean()

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

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[Mapping[str, LRScheduler | int | TrainingMode]]]:
        opt = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sched = {
            "scheduler": CosineAnnealingWarmRestarts(
                optimizer=opt, T_0=self.lr_initial_restart, T_mult=self.lr_restart_mult
            ),
            "interval": self.lr_sched_interval.name,
            "frequency": self.lr_sched_freq,
        }
        return [opt], [sched]

    @implements(pl.LightningModule)
    def predict_step(
        self, batch: Tensor | NamedSample, batch_idx: int, dataloader_idx: int | None = None
    ) -> Tensor:
        x = batch if isinstance(batch, Tensor) else batch.x
        mean, std = self(x)
        variational_dist = self._dist(mean, std)
        return torch.stack([variational_dist.mean, variational_dist.stddev], dim=-1)

    @implements(pl.LightningModule)
    def training_step(self, batch, batch_idx) -> Tensor:
        mean, std = self(batch.x)
        variational_dist = self._dist(mean, std)
        loss = self._get_loss(variational_dist, batch)
        self.train_mse(variational_dist.mean, batch.y)
        self.train_mae(variational_dist.mean, batch.y)
        results_dict = {
            f"{Stage.fit.value}/loss": float(loss.item()),  # type: ignore
            f"{Stage.fit.value}/mse": self.train_mse,
            f"{Stage.fit.value}/mae": self.train_mae,
        }
        self.log_dict(results_dict)
        return loss

    @implements(pl.LightningModule)
    def validation_epoch_end(self, outputs: list[ValStepOut]) -> None:
        targets = torch.cat([step_output.targets for step_output in outputs], 0)
        predicted_means = torch.cat([step_output.pred_means for step_output in outputs], 0)
        predicted_stddevs = torch.cat([step_output.pred_stddevs for step_output in outputs], 0)
        # squared error
        errors = ((predicted_means - targets) ** 2).detach().cpu()
        # Use an acceptable error threshold of 1 degree
        thresh = 1.0
        # Get all metrics
        f_auc, f95, _ = f_beta_metrics(
            errors=errors, uncertainty=predicted_stddevs.detach().cpu(), threshold=thresh, beta=1.0
        )
        results_dict = dict(f_auc=f_auc, f95=f95)
        results_dict = {
            f"{Stage.validate.value}/{key}": value for key, value in results_dict.items()
        }
        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    def validation_step(self, batch: BinarySample, batch_idx: int) -> ValStepOut:
        mean, std = self(batch.x)
        variational_dist = self._dist(mean, std)
        loss = self._get_loss(variational_dist, batch)
        self.train_mse(variational_dist.mean, batch.y)
        self.train_mae(variational_dist.mean, batch.y)
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

    @implements(nn.Module)
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.feature_extractor(x)
        mean_std: Tensor = self.mean_std_net(z)
        return mean_std[:, 0].sigmoid(), mean_std[:, 1].sigmoid() + torch.finfo(torch.float32).eps
