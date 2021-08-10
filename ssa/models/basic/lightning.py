#!/usr/bin/env python3
from __future__ import annotations

from bolts.data import BinarySample, PBDataModule
from bolts.models import ModelBase
from bolts.structures import MetricDict, Stage
from kit.torch import TrainingMode
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from torch import Tensor, nn
import torch.distributions as td

from ssa.weather.assessment import f_beta_metrics

from ..due import ActivationFn, FCResNet

__all__ = ["SimpleRegression"]


class SimpleRegression(ModelBase):
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
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            lr_initial_restart=lr_initial_restart,
            lr_restart_mult=lr_restart_mult,
            lr_sched_interval=lr_sched_interval,
            lr_sched_freq=lr_sched_freq,
        )
        self.activation_fn = activation_fn
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.n_power_iterations = n_power_iterations
        self.num_features = num_features
        self.snorm_coeff = snorm_coeff

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
        self.log(f"{stage.value}/val_loss", loss.item())  # type: ignore
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
        results_dict = {f"{stage.value}/{key}": value for key, value in results_dict.items()}
        results_dict["preds_mean"] = self.target_scaler.inverse_transform(
            predicted_means.detach().cpu()
        )
        results_dict["preds_std"] = predicted_stddevs.detach().cpu()
        if stage is Stage.test:
            self.results_dict = results_dict
        return results_dict

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.feature_extractor(x)
        mean_std: Tensor = self.mean_std_net(z)
        return mean_std[:, 0].sigmoid(), mean_std[:, 1].sigmoid() + torch.finfo(torch.float32).eps
