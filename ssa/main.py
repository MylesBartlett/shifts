from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional

from bolts.structures import Stage
import hydra
from hydra.utils import instantiate, to_absolute_path
from kit.hydra import SchemaRegistration
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, MISSING, OmegaConf
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from ssa.hydra.pytorch_lightning.trainer.configs import TrainerConf
from ssa.hydra.ssa.models.configs import DUEConf, SimpleRegressionConf
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
    group.add_option(name="simple", config_class=SimpleRegressionConf)

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
    es_callback = EarlyStopping(
        monitor=f"{Stage.validate.value}/val_loss",
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode="min",
    )
    ckpt_callback = ModelCheckpoint(monitor=f"{Stage.validate.value}/val_loss", mode="min")
    cfg.trainer.callbacks += [ckpt_callback, es_callback]
    print("Fitting model.")
    cfg.trainer.fit(model=cfg.model, datamodule=cfg.data)
    cfg.trainer.test(datamodule=cfg.data)
    produce_submission(
        cfg.model.results_dict["preds_mean"],
        cfg.model.results_dict["preds_std"],
        Path("./submission"),
    )
    exp_logger.experiment.finish()


def produce_submission(
    pred_mean: npt.NDArray[np.float32], pred_std: npt.NDArray[np.float32], results_dir: Path
) -> None:
    if not results_dir.exists():
        results_dir.mkdir()
    ids = np.arange(1, len(pred_mean) + 1)
    df = pd.DataFrame({"ID": ids, "PRED": pred_mean, "UNCERTAINTY": pred_std})
    df.to_csv(f"./{results_dir}/out.csv", index=False)


if __name__ == "__main__":
    launcher()
