from pathlib import Path

from ssa.weather.assessment import *


def test_evaluation():
    model_out = pd.read_csv(
        Path('./dir/df_submission.tar.gz'),
        header=None,
        names=["ID", "PRED", "UNCERTAINTY"],
        skiprows=1,
    ).iloc[:-1]
    preds = model_out["PRED"].to_numpy()
    uncert = model_out["UNCERTAINTY"].to_numpy()

    test = pd.read_csv(
        "./dir/data.tar.gz",
        header=None,
        names=["ID", "fact_temperature", "climate"],
        skiprows=1,
    ).iloc[:-1]
    test_tgt = test["fact_temperature"].to_numpy()
    pd.factorize(test["climate"])[0]

    assert len(preds) == len(test)

    shifts_vals = {
        "SCORE": 1.2266435123,
        "RMSE": 1.8421319366,
        "MAE": 1.3612728273,
        "AUC-F1": 0.5220280162,
        "F1@95%": 0.6583114832,
        "ROC-AUC": 62.961930039,
    }

    rmse = calc_rmse(preds, test_tgt)
    np.testing.assert_almost_equal(rmse, shifts_vals["RMSE"])

    mae = calc_mae(preds, test_tgt)
    np.testing.assert_almost_equal(mae, shifts_vals["MAE"])

    err = calc_rmse(preds, test_tgt, raw=True)
    f_auc, f95, _ = f_beta_metrics(err, uncert, threshold=1.0)

    np.testing.assert_almost_equal(f_auc, shifts_vals["AUC-F1"])
    np.testing.assert_almost_equal(f95, shifts_vals["F1@95%"])
