"""Base Model."""
from __future__ import annotations
from abc import abstractmethod
from typing import Mapping, NamedTuple

from conduit.data import BinarySample, CdtDataModule, NamedSample
from conduit.types import LRScheduler, Stage
import pytorch_lightning as pl
from ranzen import implements
from ranzen.torch import TrainingMode
import torch
from torch import Tensor, nn, optim
import torch.distributions as td
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics import MeanAbsoluteError, MeanSquaredError

__all__ = ["BaseVariationalModel", "ShiftsBaseModel"]

from ssa.weather.assessment import f_beta_metrics


class ValStepOut(NamedTuple):
    pred_means: Tensor
    pred_stddevs: Tensor
    targets: Tensor


class ShiftsBaseModel(pl.LightningModule):
    @abstractmethod
    def build(self, datamodule: CdtDataModule, trainer: pl.Trainer) -> None:
        ...

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
    @abstractmethod
    def predict_step(
        self, batch: Tensor | NamedSample, batch_idx: int, dataloader_idx: int | None = None
    ) -> Tensor:
        ...

    @implements(pl.LightningModule)
    @abstractmethod
    def validation_step(self, batch: BinarySample, batch_idx: int) -> ValStepOut:
        ...

    @implements(pl.LightningModule)
    def validation_epoch_end(self, outputs: list[ValStepOut]) -> None:
        pred_means = torch.cat([step_output.pred_means for step_output in outputs], 0)
        pred_stddevs = torch.cat([step_output.pred_stddevs for step_output in outputs], 0)
        targets = torch.cat([step_output.targets for step_output in outputs], 0)
        # squared error
        errors = ((pred_means - targets) ** 2).detach().cpu()
        # Use an acceptable error threshold of 1 degree
        thresh = 1.0
        # Get all metrics
        f_auc, f95, _ = f_beta_metrics(
            errors=errors, uncertainty=pred_stddevs.detach().cpu(), threshold=thresh, beta=1.0
        )
        results_dict = dict(f_auc=f_auc, f95=f95)
        results_dict = {
            f"{Stage.validate.value}/{key}": value for key, value in results_dict.items()
        }
        self.log_dict(results_dict)


class BaseVariationalModel(ShiftsBaseModel):
    def __init__(
        self,
        lr: float = 3.0e-4,
        lr_initial_restart: int = 10,
        lr_restart_mult: int = 2,
        lr_sched_interval: TrainingMode = TrainingMode.epoch,
        lr_sched_freq: int = 1,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_initial_restart = lr_initial_restart
        self.lr_restart_mult = lr_restart_mult
        self.lr_sched_interval = lr_sched_interval
        self.lr_sched_freq = lr_sched_freq
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.train_mse = MeanSquaredError(squared=False)
        self.val_mse = MeanSquaredError(squared=False)

    @abstractmethod
    def _get_loss(self, variational_dist: td.Distribution, batch: BinarySample):
        ...

    @abstractmethod
    def _inference_step(self, x: Tensor, batch_idx: int, dataloader_idx: int | None = None):
        ...

    @implements(nn.Module)
    @abstractmethod
    def forward(self, x: Tensor) -> td.Distribution:
        ...

    @implements(ShiftsBaseModel)
    def predict_step(
        self, batch: Tensor | NamedSample, batch_idx: int, dataloader_idx: int | None = None
    ) -> Tensor:
        x = batch if isinstance(batch, Tensor) else batch.x
        return self._inference_step(x=x, batch_idx=batch_idx, dataloader_idx=dataloader_idx)

    @implements(pl.LightningModule)
    def training_step(self, batch, batch_idx) -> Tensor:
        variational_dist = self.forward(batch.x)
        loss = self._get_loss(variational_dist, batch)
        self.train_mse(variational_dist.mean.detach(), batch.y)
        self.train_mae(variational_dist.mean.detach(), batch.y)
        results_dict = {
            f"{Stage.fit.value}/loss": float(loss.item()),  # type: ignore
            f"{Stage.fit.value}/mse": self.train_mse,
            f"{Stage.fit.value}/mae": self.train_mae,
        }
        self.log_dict(results_dict)
        return loss
