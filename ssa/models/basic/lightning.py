#!/usr/bin/env python3
from __future__ import annotations

from bolts.common import MetricDict, Stage
from bolts.data import BinarySample, PBDataModule
from bolts.models import ModelBase
from kit.torch import TrainingMode
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from torch import Tensor, nn
import torch.distributions as td

from ssa.weather.assessment import f_beta_metrics

__all__ = ["SimpleRegression"]


class SimpleRegression(ModelBase):
    net: nn.Module
    mean_net: nn.Module
    std_net: nn.Module

    def __init__(
        self,
        weight_decay: float = 0.0,
        lr: float = 3.0e-4,
        lr_initial_restart: int = 10,
        lr_restart_mult: int = 2,
        lr_sched_interval: TrainingMode = TrainingMode.epoch,
        lr_sched_freq: int = 1,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            lr_initial_restart=lr_initial_restart,
            lr_restart_mult=lr_restart_mult,
            lr_sched_interval=lr_sched_interval,
            lr_sched_freq=lr_sched_freq,
        )
        self.relu = nn.ReLU()

    def build(self, datamodule: PBDataModule, trainer: pl.Trainer) -> None:
        self.net = nn.Sequential(
            nn.Linear(datamodule.dim_x[0], 1_000),
            nn.SiLU(),
            nn.Linear(1_000, 1_000),
            nn.SiLU(),
        )
        self.mean_net = nn.Sequential(nn.Linear(1_000, 1), nn.Sigmoid())
        self.std_net = nn.Sequential(nn.Linear(1_000, 1), nn.ReLU())

        self.feature_scaler = datamodule.feature_transform
        self.target_scaler = datamodule.target_transform

    def training_step(self, batch, batch_idx) -> Tensor:
        mean, std = self(batch.x)
        out_dist = td.Normal(mean, std)
        return -out_dist.log_prob(batch.y).mean()

    def _inference_step(self, batch: BinarySample, *, stage: Stage) -> STEP_OUTPUT:
        mean, std = self(batch.x)
        variational_dist = td.Normal(mean, std)
        loss = -variational_dist.log_prob(batch.y).mean()
        self.log_dict({f"{stage}/expected_log_prob": loss.item()})  # type: ignore
        return {
            "y": batch.y.view(-1),
            "predicted_means": variational_dist.mean,
            "predicted_stddevs": variational_dist.stddev,
        }

    def _inference_epoch_end(self, outputs: EPOCH_OUTPUT, stage: Stage) -> MetricDict:
        targets = torch.cat([step_output["y"] for step_output in outputs], 0)
        predicted_means = torch.cat([step_output["predicted_means"] for step_output in outputs], 0)
        predicted_stddevs = torch.cat(
            [step_output["predicted_stddevs"] for step_output in outputs], 0
        )
        # squared error
        errors = ((predicted_means - targets) ** 2).detach().cpu()
        # Use an acceptable error threshold of 1 degree
        thresh = 1.0
        # Get all metrics
        f_auc, f95, _ = f_beta_metrics(
            errors=errors, uncertainty=predicted_stddevs.detach().cpu(), threshold=thresh, beta=1.0
        )
        results_dict = dict(f_auc=f_auc, f95=f95)
        results_dict = {f"{stage}/{key}": value for key, value in results_dict.items()}
        results_dict["preds_mean"] = self.target_scaler.inverse_transform(
            predicted_means.detach().cpu()
        )
        results_dict["preds_std"] = predicted_stddevs.detach().cpu()
        if stage == "test":
            self.results_dict = results_dict
        return results_dict

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.net(x)
        mean: Tensor = self.mean_net(z)
        std: Tensor = self.std_net(z) + torch.finfo(torch.float32).eps
        return mean.squeeze(), std.squeeze()
