import datetime
from typing import Tuple

import pandas as pd
import numpy as np
from numpy import ndarray
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor

from preprocess import _mode_of_freq, _src_tgt_split


def recurrent_inference(
                    rec_freq: int,
                    model: object,
                    data: pd.DataFrame,
                    seq: int,
                    d_model: int,
                    dilation: int,
                    src_tgt_seq: Tuple[int, int],
                    ) -> pd.Series:
    """再帰的に推論を行う
    引数: 
        rec_freq: 推論回数
        model: 訓練済みモデル
        data: オリジナルデータ
        seq: 訓練条件の seq
        d_model: 訓練条件の d_model
        dilation: 訓練条件の dilation
        src_tgt_seq: 訓練条件の src_tgt_seq,
    """
    sum_freq = _mode_of_freq(data).item_cnt_day
    start_idx = sum_freq.index[-1] + datetime.timedelta(1)
    scs = StandardScaler().fit(sum_freq.values.reshape(-1, 1))

    src_seq, tgt_seq = src_tgt_seq
    inference_seq = np.array([])
    for _ in range(rec_freq):
        embed = _tde_for_inference(sum_freq, seq, d_model, dilation)
        src, tgt = _src_tgt_split(embed, src_seq, tgt_seq)
        output = _inference(model, src, tgt).reshape(-1)
        inversed = scs.inverse_transform(output[-1].reshape(-1, 1)).reshape(-1)
        sum_freq = np.append(sum_freq, np.round(inversed))
        inference_seq = np.append(inference_seq, np.round(inversed))

    end_idx = start_idx + datetime.timedelta(len(inference_seq) - 1)
    index = pd.date_range(start_idx, end_idx)
    return pd.Series(inference_seq, index)


def _inference(model:    object, src: Tensor, tgt: Tensor) -> Tensor:
    src = torch.from_numpy(src.astype(np.float32)).T.unsqueeze(0)
    tgt = torch.from_numpy(tgt.astype(np.float32)).T.unsqueeze(0)
    model.eval()
    output = model(src, tgt).detach().numpy()
    return output


def _tde_for_inference(
                    ds: pd.Series,
                    seq: int,
                    d_model: int,
                    dilation: int
                    ) -> ndarray:
    for_array = []
    for i in range(d_model):
        if i != 0:
            for_array.append(ds[-seq - i * dilation: -i * dilation])
        else:
            for_array.append(ds[-seq:])
    time_delay_embedded = np.array([content for content in reversed(for_array)])
    return time_delay_embedded
