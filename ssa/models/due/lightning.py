"""PyTorch-Lightning wrapper for DUE."""
from __future__ import annotations
from typing import Mapping, NamedTuple

from bolts.data.datamodules.base import PBDataModule
from bolts.data.structures import BinarySample, NamedSample
from bolts.structures import LRScheduler, Stage
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from kit import implements
from kit.decorators import implements, parsable
from kit.torch import TrainingMode
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data.dataset import Subset

from ssa.weather.assessment import f_beta_metrics

from .dkl import DKLGP, GPKernel, get_initial_inducing_points, get_initial_lengthscale
from .fc_resnet import ActivationFn, FCResNet

__all__ = ["DUE"]


class ValStepOut(NamedTuple):
    pred_means: Tensor
    pred_stddevs: Tensor
    targets: Tensor


class DUE(pl.LightningModule):
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
        if num_inducing_points >= num_inducing_point_refs:
            raise ValueError(
                f"NUmber of reference points used to compute the initial inducing points must "
                f"exceed the number of inducing points. Received 'num_inducing_points={num_inducing_points}' "
                f"and `num_inducing_point_refs={num_inducing_point_refs}.`"
            )
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_initial_restart = lr_initial_restart
        self.lr_restart_mult = lr_restart_mult
        self.lr_sched_interval = lr_sched_interval
        self.lr_sched_freq = lr_sched_freq

        self.num_features = num_features
        self.depth = depth
        self.snorm_coeff = snorm_coeff
        self.n_power_iterations = n_power_iterations
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn

        self.num_inducing_points = num_inducing_points
        self.num_inducing_point_refs = num_inducing_point_refs
        self.kernel = kernel
        self.beta = beta

        self.likelihood = GaussianLikelihood()

    def _get_loss(self, variational_dist: MultivariateNormal, *, batch: BinarySample) -> Tensor:
        return -self.loss_fn(variational_dist, target=batch.y)

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
        print(f"Extracting features from a subset of {self.num_inducing_point_refs} data-points.")
        fe_lm = FeatureExtractorLM(self.feature_extractor)
        trainer.test(
            fe_lm,
            test_dataloaders=train_dl,
            verbose=False,
        )
        datamodule._train_data = train_data_full

        features = fe_lm.features.cpu()
        print(f"Computing initial inducing points for GP.")
        initial_inducing_points = get_initial_inducing_points(
            f_X_sample=features.numpy(), num_inducing_points=self.num_inducing_points
        )
        print(f"Computing initial lengthscale for GP.")
        initial_lengthscale = get_initial_lengthscale(f_X_samples=features)

        self.dklgp = DKLGP(
            feature_extractor=self.feature_extractor,
            num_outputs=datamodule.dim_y[0],
            initial_inducing_points=initial_inducing_points,
            initial_lengthscale=initial_lengthscale,
            kernel=self.kernel,
        )
        self.loss_fn = VariationalELBO(
            likelihood=self.likelihood,
            model=self.dklgp.gp,
            num_data=datamodule.num_train_samples,
            beta=self.beta,
        )

        self.feature_scaler = datamodule.feature_transform
        self.target_scaler = datamodule.target_transform

    def inference_step(
        self, batch: Tensor | NamedSample, batch_idx: int, dataloader_idx: int | None = None
    ) -> Tensor:
        x = batch if isinstance(batch, Tensor) else batch.x
        variational_dist = self.dklgp(x)
        return torch.stack([variational_dist.mean, variational_dist.stddev], dim=-1)

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
        return self.inference_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)

    @implements(pl.LightningModule)
    def training_step(self, batch: BinarySample, batch_idx: int) -> Tensor:
        logits = self.forward(batch.x)
        loss = self._get_loss(variational_dist=logits, batch=batch)
        self.log_dict({f"{Stage.fit.value}/loss": float(loss.item())})  # type: ignore
        return loss

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
            errors=errors, uncertainty=pred_stddevs, threshold=thresh, beta=1.0
        )
        results_dict = dict(f_auc=f_auc, f95=f95)
        results_dict = {
            f"{Stage.validate.value}/{key}": value for key, value in results_dict.items()
        }
        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    def validation_step(self, batch: BinarySample, batch_idx: int) -> ValStepOut:
        x = batch if isinstance(batch, Tensor) else batch.x
        variational_dist = self.dklgp(x)

        loss = self.likelihood.expected_log_prob(input=variational_dist, target=batch.y).mean()
        self.log_dict({f"{Stage.validate.value}/expected_log_prob": loss.item()})  # type: ignore

        return ValStepOut(
            pred_means=variational_dist.mean,
            pred_stddevs=variational_dist.stddev,
            targets=batch.y,
        )

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
