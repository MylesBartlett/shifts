from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
from hydra.utils import instantiate, to_absolute_path
from kit.hydra import SchemaRegistration
from omegaconf import MISSING, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch

from ssa.hydra.pytorch_lightning.trainer.configs import TrainerConf
from ssa.hydra.ssa.models.configs import DUEConf
from ssa.hydra.ssa.weather.data.configs import WeatherDataModuleConf
from ssa.submission import Prediction, produce_submission


@dataclass
class ExpConfig:
    log_offline: bool = False
    results_dir: str = "results"
    seed: int = 42


@dataclass
class Experiment:
    """Configuration for this program.

    The values can be changed via yaml files and commandline arguments.
    """

    _target_: str = "ssa.main.Experiment"
    data: Any = MISSING
    exp: ExpConfig = MISSING
    exp_group: str = "Testing"
    model: Any = MISSING
    trainer: Any = MISSING

    def start(self, raw_config: Optional[Dict[str, Any]]) -> None:
        """Script entrypoint."""
        print(f"Current working directory: '{os.getcwd()}'")
        print("-----\n" + str(raw_config) + "\n-----")

        exp_logger = WandbLogger(
            entity="predictive-analytics-lab",
            project="shifts",
            offline=self.exp.log_offline,
            group=self.exp_group,
            reinit=True,  # for multirun compatibility
        )

        if raw_config is not None:
            exp_logger.log_hyperparams(raw_config)
        self.trainer.logger = exp_logger
        pl.seed_everything(self.exp.seed)

        self.data.root = to_absolute_path(self.data.root)
        print("Preparing the data.")
        self.data.prepare_data()
        self.data.setup()
        # Build the model
        print("Building model.")
        self.model.build(datamodule=self.data, trainer=self.trainer)
        # Fit the model
        print("Fitting model.")
        self.trainer.fit(model=self.model, datamodule=self.data)
        # results_dict = self.trainer.test(model=self.model, datamodule=self.data)
        preds_with_unc_ls = self.trainer.predict(
            model=self.model, dataloaders=self.data.test_dataloader()
        )
        preds_with_unc_cat = torch.cat(preds_with_unc_ls, dim=0)
        preds, uncertainty = torch.chunk(preds_with_unc_cat, chunks=2, dim=-1)
        preds = self.data.target_transform.inverse_transform(preds)
        preds_dc = Prediction(pred=preds.cpu().numpy(), uncertainty=uncertainty.cpu().numpy())

        produce_submission(predictions=preds_dc, results_dir=Path(self.exp.results_dir))
        exp_logger.experiment.finish()


sr = SchemaRegistration()
sr.register(path="main_schema", config_class=Experiment)

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
def launcher(hydra_config: Experiment) -> None:
    """Instantiate with hydra and get the experiments running!"""
    cfg: Experiment = instantiate(hydra_config, _recursive_=True)
    cfg.start(raw_config=OmegaConf.to_container(hydra_config, resolve=True, enum_to_str=True))


if __name__ == "__main__":
    launcher()
