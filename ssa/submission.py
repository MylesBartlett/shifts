from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["Prediction", "produce_submission"]


LOGGER = logging.getLogger(__file__)


@dataclass
class Prediction:
    pred: npt.NDArray[np.float32]
    uncertainty: npt.NDArray[np.float32]


def produce_submission(predictions: Prediction, results_dir: Path) -> None:
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    ids = np.arange(1, len(predictions.pred) + 1)
    df = pd.DataFrame(
        {
            "ID": ids,
            "PRED": predictions.pred.squeeze(),
            "UNCERTAINTY": predictions.uncertainty.squeeze(),
        }
    )
    submission_fp = results_dir / "submission.csv"
    df.to_csv(submission_fp, index=False)
    LOGGER.info(f"Submission saved to {submission_fp}")
