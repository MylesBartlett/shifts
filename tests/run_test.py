from __future__ import annotations
from pathlib import Path
from typing import Final

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest

from ssa.main import Experiment

CFG_PTH: Final[str] = "../ssa/configs"
SCHEMAS: Final[list[str]] = [
    "exp=unit_test",
    "trainer=unit_test",
    "model=due_test",
]
BATCH_SIZE = [
    "data.train_batch_size=10",
    "data.eval_batch_size=25",
]


@pytest.mark.parametrize("datamodule", ["weather"])
def test_with_initialize(datamodule: str) -> None:
    """Quick run on models to check nothing's broken."""
    data_dir = Path("~/Data").expanduser()
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="main",
            overrides=[f"data={datamodule}", f"data.root={data_dir}"] + SCHEMAS + BATCH_SIZE,
        )
        cfg: Experiment = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        cfg.start(raw_config=OmegaConf.to_container(hydra_cfg, resolve=True, enum_to_str=True))


@pytest.mark.parametrize("datamodule", ["weather"])
@pytest.mark.parametrize("model", ["simple", "due_test"])
def test_regressor(datamodule: str, model: str) -> None:
    """Quick run on models to check nothing's broken."""
    data_dir = Path("~/Data").expanduser()
    with initialize(config_path=CFG_PTH):
        hydra_cfg = compose(
            config_name="main",
            overrides=SCHEMAS
            + [f"data={datamodule}", f"data.root={data_dir}", f"model={model}"]
            + BATCH_SIZE,
        )
        cfg: Experiment = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        cfg.start(raw_config=OmegaConf.to_container(hydra_cfg, resolve=True, enum_to_str=True))
