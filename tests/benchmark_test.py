import timeit

import numpy as np
from numpy.random import randn
import pandas as pd
import torch


def generate_data(n):
    return pd.DataFrame(
        {
            "a": randn(n),
            "b": randn(n),
            "c": randn(n),
        }
    )


def benchmark(df, name, saver, loader):
    verify(df, loader, saver)
    save_timer = timeit.Timer(lambda: saver(df))
    load_timer = timeit.Timer(lambda: loader().a.sum())
    save_n, save_time = save_timer.autorange()
    load_n, load_time = load_timer.autorange()
    print(
        f"{name:<15s} : "
        f"{save_n / save_time:>20.3f} save/s : "
        f"{load_n / load_time:>20.3f} load/s : "
    )


def verify(df, loader, saver):
    saver(df)
    loaded = loader()
    assert np.allclose(loaded.a.sum(), df.a.sum())
    assert np.allclose(loaded.b.sum(), df.b.sum())
    assert list(loaded.columns) == list(df.columns), loaded.columns


def save_feather(df):
    df = df.reset_index()
    df.to_feather("dummy.feather")


def load_feather():
    df = pd.read_feather("dummy.feather")
    df = df.drop(columns=["index"])
    return df


def save_numpy(df: pd.DataFrame):
    nparray = df.to_numpy()
    np.save("dummy", nparray)


def load_numpy():
    npzfile = np.load("dummy.npy")
    return pd.DataFrame(npzfile, columns=["a", "b", "c"])


def main():
    for num_samples in (1 * 10 ** exp for exp in range(4, 9)):
        print(f"{num_samples=:_}")
        df = generate_data(num_samples)
        benchmark(df, "numpy", save_numpy, load_numpy)
        benchmark(df, "dummy", lambda df: None, lambda: df)
        benchmark(
            df,
            "csv",
            lambda df: df.to_csv("dummy.csv", index=False),
            lambda: pd.read_csv("dummy.csv", index_col=None),
        )
        benchmark(
            df,
            "pickle",
            lambda df: df.to_pickle("dummy.pickle"),
            lambda: pd.read_pickle("dummy.pickle"),
        )
        benchmark(
            df,
            "torch",
            lambda df: torch.save(torch.tensor(df.values), "dummy.torch"),
            lambda: pd.DataFrame(torch.load("dummy.torch").numpy(), columns=["a", "b", "c"]),
        )
        benchmark(df, "feather", save_feather, load_feather)
        benchmark(
            df,
            "parquet",
            lambda df: df.to_parquet("dummy.parquet", allow_truncated_timestamps=True),
            lambda: pd.read_parquet("dummy.parquet"),
        )


if __name__ == "__main__":
    main()
