import numpy as np
import pandas as pd

from typing import List, Tuple


def mode_of_freq(data: pd.DataFrame,
                 key='date',
                 freq='D',
                 mode='sum'
                 ) -> pd.DataFrame:
    # 時系列(key)について日毎(D)の売上数の合計値(sum)で出力
    mode_of_key = getattr(data.groupby(pd.Grouper(key=key, freq=freq)), mode)

    return mode_of_key()


def making_dataset(
        ds: pd.Series,
        span=7,
        train_rate=0.9)-> list[tuple[np.ndarray, np.ndarray]]:
    data = ds.copy()
    endpoint = len(data) - span
    # [(入力データ, 正解データ)...]
    x_data = np.stack([data[i: i+span] for i in range(0, endpoint)])
    y_data = np.stack([data[i+span] for i in range(0, endpoint)])

    size = int(len(data)*0.9)
    x_train = x_data[:size, :]
    x_test = x_data[size:, :]
    y_train = y_data[:size]
    y_test = y_data[size:]

    return x_train, x_test, y_train, y_test