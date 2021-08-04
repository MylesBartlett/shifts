from dataclasses import dataclass
import os
from typing import Any, Dict, Final, Optional

import hydra
from hydra.utils import instantiate, to_absolute_path
from kit.hydra import SchemaRegistration
from omegaconf import DictConfig, MISSING, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from ssa.hydra.pytorch_lightning.trainer.configs import TrainerConf
from ssa.hydra.ssa.models.configs import DUEConf
from ssa.hydra.ssa.weather.data.configs import WeatherDataModuleConf


@dataclass
class ExpConfig:
    log_offline: bool = False
    results_dir: Optional[str] = None
    seed: int = 42


@dataclass
class Config:
    """Configuration for this program.

    The values can be changed via yaml files and commandline arguments.
    """

    _target_: str = "ssa.main.Config"
    data: Any = MISSING
    exp: ExpConfig = MISSING
    exp_group: str = "Testing"
    model: Any = MISSING
    trainer: Any = MISSING


sr = SchemaRegistration()
sr.register(path="main_schema", config_class=Config)

# 'datamodule' group
with sr.new_group(group_name="schema/data", target_path="data") as group:
    group.add_option(name="weather", config_class=WeatherDataModuleConf)

# 'model' group
with sr.new_group(group_name="schema/model", target_path="model") as group:
    group.add_option(name="due", config_class=DUEConf)

# 'trainer' group
with sr.new_group(group_name="schema/trainer", target_path="trainer") as group:
    group.add_option(name="trainer", config_class=TrainerConf)


@hydra.main(config_path="configs", config_name="main")
def launcher(hydra_config: DictConfig) -> None:
    """Instantiate with hydra and get the experiments running!"""
    if hasattr(hydra_config.data, "root"):
        hydra_config.data.root = to_absolute_path(hydra_config.data.root)
    cfg: Config = instantiate(hydra_config, _recursive_=True, _convert_="partial")
    start(cfg, raw_config=OmegaConf.to_container(hydra_config, resolve=True, enum_to_str=True))


def start(cfg: Config, raw_config: Optional[Dict[str, Any]]) -> None:
    """Script entrypoint."""
    print(f"Current working directory: '{os.getcwd()}'")
    print("-----\n" + str(raw_config) + "\n-----")

    exp_logger = WandbLogger(
        entity="predictive-analytics-lab",
        project="shifts",
        offline=cfg.exp.log_offline,
        group=cfg.exp_group,
        reinit=True,  # for multirun compatibility
    )

    if raw_config is not None:
        exp_logger.log_hyperparams(raw_config)
    cfg.trainer.logger = exp_logger
    pl.seed_everything(cfg.exp.seed)

    print("Preparing the data.")
    cfg.data.prepare_data()
    cfg.data.setup()
    # Build the model
    print("Building model.")
    cfg.model.build(datamodule=cfg.data, trainer=cfg.trainer)
    # Fit the model
    print("Fitting model.")
    cfg.trainer.fit(model=cfg.model, datamodule=cfg.data)
    exp_logger.experiment.finish()


if __name__ == "__main__":
    launcher()