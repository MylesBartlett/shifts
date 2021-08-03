from __future__ import annotations
from enum import Enum
from functools import partial
from typing import Union

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, RQKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)
from kit import str_to_enum
import numpy as np
import numpy.typing as npt
from sklearn import cluster
import torch
from torch import Tensor
import torch.nn as nn

__all__ = ["DKLGP", "GP", "get_initial_inducing_points", "get_initial_lengthscale", "GPKernel"]


def get_initial_inducing_points(
    f_X_sample: npt.NDArray[np.float32], num_inducing_points: int
) -> Tensor:
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=num_inducing_points, batch_size=num_inducing_points * 10
    )
    kmeans.fit(f_X_sample)
    initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_)

    return initial_inducing_points


def get_initial_lengthscale(f_X_samples: Tensor) -> Tensor:
    if torch.cuda.is_available():
        f_X_samples = f_X_samples.cuda()

    initial_lengthscale = torch.pdist(f_X_samples).mean()

    return initial_lengthscale.cpu()


class GPKernel(Enum):
    rbf = RBFKernel
    matern12 = partial(MaternKernel, nu=1 / 2)
    matern32 = partial(MaternKernel, nu=3 / 2)
    matern52 = partial(MaternKernel, nu=5 / 2)
    rbq = RQKernel


class GP(ApproximateGP):
    def __init__(
        self,
        num_outputs: int,
        initial_lengthscale: Tensor,
        initial_inducing_points: Tensor,
        kernel: Union[GPKernel, str] = GPKernel.rbf,
    ) -> None:
        if isinstance(kernel, str):
            kernel = str_to_enum(str_=kernel, enum=GPKernel)

        num_inducing_points = initial_inducing_points.shape[0]

        if num_outputs > 1:
            batch_shape = torch.Size([num_outputs])
        else:
            batch_shape = torch.Size([])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing_points, batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points=initial_inducing_points,
            variational_distribution=variational_distribution,
        )

        if num_outputs > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                base_variational_strategy=variational_strategy, num_tasks=num_outputs
            )

        super().__init__(variational_strategy=variational_strategy)

        kwargs = {
            "batch_shape": batch_shape,
        }

        kernel_fn = GPKernel.value(**kwargs)

        kernel_fn.lengthscale = initial_lengthscale * torch.ones_like(kernel_fn.lengthscale)

        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(kernel, batch_shape=batch_shape)

    def forward(self, x) -> MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)

        return MultivariateNormal(mean, covar)

    @property
    def inducing_points(self) -> nn.Parameter | None:
        for name, param in self.named_parameters():
            if "inducing_points" in name:
                return param


class DKLGP(gpytorch.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        num_outputs: int,
        initial_inducing_points: Tensor,
        initial_lengthscale: Tensor,
        kernel: Union[GPKernel, str] = GPKernel.rbf,
    ) -> None:
        """
        This wrapper class is necessary because ApproximateGP (above) does some magic
        on the forward method which is not compatible with a feature_extractor.
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        initial_lengthscale
        self.gp = GP(
            num_outputs=num_outputs,
            kernel=kernel,
            initial_inducing_points=initial_inducing_points,
            initial_lengthscale=initial_lengthscale,
        )

    def forward(self, x: Tensor) -> MultivariateNormal:
        features = self.feature_extractor(x)
        return self.gp(features)
