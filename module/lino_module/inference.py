import pandas as pd
import numpy as np
import torch

from preprocess import _mode_of_freq, _src_tgt_split


def inference(model, data, seq, d_model, dilation, src_tgt_seq):
    src_seq, tgt_seq = src_tgt_seq
    df = _mode_of_freq(data).item_cnt_day.values
    df = _tde_for_inference(df, seq, d_model, dilation)
    src, tgt = _src_tgt_split(df, src_seq, tgt_seq)

    src = torch.from_numpy(src.astype(np.float32)).T.unsqueeze(0)
    tgt = torch.from_numpy(tgt.astype(np.float32)).T.unsqueeze(0)
    model.eval()
    return model(src, tgt)


def _tde_for_inference(ds: pd.Series, seq, d_model, dilation) -> np.ndarray:
    for_array = []
    for i in range(d_model):
        if i != 0:
            for_array.append(ds[-seq - i * dilation: -i * dilation])
        else:
            for_array.append(ds[-seq:])
    time_delay_embedded = np.array([content for content in reversed(for_array)])
    return time_delay_embedded
