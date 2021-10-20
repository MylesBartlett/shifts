from enum import Enum
from typing import Optional, Union

from ranzen import str_to_enum
import torch.nn as nn

from .layers import spectral_norm_fc

__all__ = ["FCResNet", "ActivationFn"]


class ActivationFn(Enum):
    relu = nn.ReLU
    lrelu = nn.LeakyReLU
    elu = nn.ELU
    gelu = nn.GELU


class FCResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_features: int = 128,
        depth: int = 4,
        spectral_normalization: bool = True,
        snorm_coeff: float = 0.95,
        n_power_iterations: int = 1,
        dropout_rate: float = 0.01,
        num_outputs: Optional[int] = None,
        activation_fn: Union[ActivationFn, str] = ActivationFn.relu,
    ) -> None:
        super().__init__()
        """
        ResFNN architecture

        Introduced in SNGP: https://arxiv.org/abs/2006.10108
        """
        self.first = nn.Linear(in_channels, num_features)
        self.residuals = nn.ModuleList(
            [nn.Linear(num_features, num_features) for _ in range(depth)]
        )
        self.dropout = nn.Dropout(dropout_rate)

        if spectral_normalization:
            self.first = spectral_norm_fc(
                self.first, coeff=snorm_coeff, n_power_iterations=n_power_iterations
            )

            for i in range(len(self.residuals)):
                self.residuals[i] = spectral_norm_fc(
                    self.residuals[i],
                    coeff=snorm_coeff,
                    n_power_iterations=n_power_iterations,
                )

        self.num_outputs = num_outputs
        if num_outputs is not None:
            self.last = nn.Linear(num_features, num_outputs)

        if isinstance(activation_fn, str):
            activation_fn = str_to_enum(str_=activation_fn, enum=ActivationFn)
        self.activation_fn = activation_fn.value()

    def forward(self, x):
        x = self.first(x)

        for residual in self.residuals:
            x = x + self.dropout(self.activation_fn(residual(x)))

        if self.num_outputs is not None:
            x = self.last(x)

        return x
