from __future__ import annotations
from typing import Final

from configen.config import Config
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest

from ssa.main import start

CFG_PTH: Final[str] = "../ssa/configs"
SCHEMAS: Final[list[str]] = [
    "exp=unit_test",
    "trainer=unit_test",
    "model=due_test",
]


@pytest.mark.parametrize("dm", ["weather"])
def test_with_initialize(dm: str) -> None:
    """Quick run on models to check nothing's broken."""
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="main",
            overrides=[f"data={dm}"] + SCHEMAS,
        )
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")

        start(cfg, raw_config=OmegaConf.to_container(hydra_cfg, resolve=True, enum_to_str=True))
