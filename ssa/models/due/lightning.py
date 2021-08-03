"""PyTorch-Lightning wrapper for DUE."""
from __future__ import annotations
from typing import Union

from bolts.common import MetricDict, Stage
from bolts.data.datamodules.base import PBDataModule
from bolts.data.structures import NamedSample, TernarySample
from bolts.models import ModelBase
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from kit import implements
from kit.decorators import implements, parsable
from kit.misc import gcopy
from kit.torch import TrainingMode
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from torch import Tensor, nn
from torch.utils.data.dataset import Subset

from ssa.motion.sdc.assessment import calc_uncertainty_regection_curve
from ssa.weather.assessment import f_beta_metrics

from .dkl import DKLGP, GPKernel, get_initial_inducing_points, get_initial_lengthscale
from .fc_resnet import ActivationFn, FCResNet

__all__ = ["DUE"]


class DUE(ModelBase):
    """
    PyTorch-Lightning implementation of Deterministic Uncertainty Estimation (DUE) as introduced in
    'On Feature Collapse and Deep Kernel Learning for Single Forward Pass Uncertainty'
    (https://arxiv.org/abs/2102.11409)
    """

    dklgp: DKLGP
    loss_fn: VariationalELBO

    @parsable
    def __init__(
        self,
        *,
        # Feature-extractor settings
        num_features: int = 128,
        depth: int = 4,
        snorm_coeff: float = 0.95,
        n_power_iterations: int = 1,
        dropout_rate: float = 0.01,
        activation_fn: ActivationFn = ActivationFn.relu,
        # GP settings
        num_inducing_points: int = 20,
        num_inducing_point_refs: int = 1000,
        kernel: GPKernel = GPKernel.matern12,
        # Weight placed on the prior
        beta: float = 1.0,
        # Training config
        weight_decay: float = 0.0,
        lr: float = 3.0e-4,
        lr_initial_restart: int = 10,
        lr_restart_mult: int = 2,
        lr_sched_interval: TrainingMode = TrainingMode.epoch,
        lr_sched_freq: int = 1,
    ) -> None:
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            lr_initial_restart=lr_initial_restart,
            lr_restart_mult=lr_restart_mult,
            lr_sched_interval=lr_sched_interval,
            lr_sched_freq=lr_sched_freq,
        )

        self.num_features = num_features
        self.depth = depth
        self.coeff = snorm_coeff
        self.n_power_iterations = n_power_iterations
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn

        self.num_inducing_points = num_inducing_points
        self.num_inducing_point_refs = num_inducing_point_refs
        self.kernel = kernel
        self.beta = beta

        self.likelihood = GaussianLikelihood()

    def _get_loss(self, variational_dist: MultivariateNormal, *, batch: TernarySample) -> Tensor:
        return -self.loss_fn(variational_dist, target=batch.y)

    def build(self, datamodule: PBDataModule, trainer: pl.Trainer) -> None:
        self.feature_extractor = FCResNet(
            in_channels=datamodule.dim_x,
            num_features=self.num_features,
            depth=self.depth,
            snorm_coeff=self.snorm_coeff,
            n_power_iterations=self.n_power_iterations,
            dropout_rate=self.dropout_rate,
            activation_fn=self.activation_fn,
        )
        # Compute the initial inducing points and initial lengthscale from a subset of
        # the training data
        train_data_full = datamodule._train_data
        train_data_subset = Subset(
            train_data_full,
            indices=torch.randperm(datamodule.num_train_samples)[
                : self.num_inducing_point_refs
            ].tolist(),
        )
        datamodule._train_data = train_data_subset

        train_dl = datamodule.train_dataloader(batch_size=datamodule.eval_batch_size)
        fe_runner = gcopy(trainer, limit_test_batches=self.num_inducing_point_refs)
        fe_lm = FeatureExtractorLM(self.feature_extractor)
        fe_runner.test(
            fe_lm,
            test_dataloaders=train_dl,
            verbose=False,
        )
        datamodule._train_data = train_data_full

        features = fe_lm.features.cpu()
        initial_inducing_points = get_initial_inducing_points(
            f_X_sample=features.numpy(), num_inducing_points=self.num_inducing_points
        )
        initial_lengthscale = get_initial_lengthscale(f_X_samples=features)

        self.dklgp = DKLGP(
            feature_extractor=self.feature_extractor,
            num_outputs=datamodule.dim_y,
            initial_inducing_points=initial_inducing_points,
            initial_lengthscale=initial_lengthscale,
            kernel=self.kernel,
        )
        self.loss_fn = VariationalELBO(
            likelihood=self.likelihood,
            model=self.dklgp,
            num_data=datamodule.num_train_samples,
            beta=self.beta,
        )

    @implements(pl.LightningModule)
    def training_step(self, batch: TernarySample, batch_idx: int) -> Tensor:
        logits = self.forward(batch.x)
        loss = self._get_loss(variational_dist=logits, batch=batch)
        self.log_dict({f"train/loss": float(loss.item())})  # type: ignore
        return loss

    @implements(ModelBase)
    def _inference_step(self, batch: TernarySample, *, stage: Stage) -> STEP_OUTPUT:
        variational_dist = self.dklgp(batch.x)
        ol = self.likelihood(variational_dist)
        loss = self.likelihood.expected_log_prob(variational_dist, batch.y)
        self.log_dict({f"{stage}/expected_log_prob": loss.item()})  # type: ignore
        return {
            "y": batch.y.view(-1),
            "predicted_means": ol.mean,
            "uncertainties": ol.std,
        }

    @implements(ModelBase)
    def _inference_epoch_end(self, outputs: EPOCH_OUTPUT, stage: Stage) -> MetricDict:
        targets = torch.cat([step_output["y"] for step_output in outputs], 0)
        predicted_means = torch.cat([step_output["predicted_means"] for step_output in outputs], 0)
        uncertainties = torch.cat([step_output["pred_stds"] for step_output in outputs], 0)
        # squared error
        errors = (predicted_means - targets) ** 2
        # MSE retention values
        rejection_mse = calc_uncertainty_regection_curve(errors, uncertainties)
        retention_mse = rejection_mse[::-1]
        # Use an acceptable error threshold of 1 degree
        thresh = 1.0
        # Get all metrics
        f_auc, f95, retention_f1 = f_beta_metrics(
            errors=errors, uncertainty=uncertainties, threshold=thresh, beta=1.0
        )
        results_dict = dict(
            retention_mse=retention_mse, f_auc=f_auc, f95=f95, retention_f1=retention_f1
        )
        results_dict = {
            f"{stage}/{self.target_name}_{key}": value for key, value in results_dict.items()
        }
        return results_dict

    @implements(nn.Module)
    def forward(self, x: Tensor) -> MultivariateNormal:
        return self.dklgp(x)


class FeatureExtractorLM(pl.LightningModule):
    """Wrapper for extractor model."""

    _features: Tensor | None

    def __init__(self, feature_extractor: nn.Module) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor

    @property
    def features(self) -> Tensor:
        if self._features is None:
            raise AttributeError("No features have been extracted.")
        return self._features

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.feature_extractor(x)

    @implements(pl.LightningModule)
    def test_step(self, batch: NamedSample, batch_idx: int) -> STEP_OUTPUT:
        return self.feature_extractor(batch.x)

    @implements(pl.LightningModule)
    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._features = torch.cat(outputs)
