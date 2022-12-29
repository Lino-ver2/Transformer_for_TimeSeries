import numpy as np
import pandas as pd


def tde_for_inference(ds: pd.Series, seq, d_model, dilation) -> np.ndarray:
    for_array = []
    for i in range(d_model):
        if i != 0:
            for_array.append(ds[-seq - i * dilation: -i * dilation])
        else:
            for_array.append(ds[-seq:])
    time_delay_embedded = np.array([content for content in reversed(for_array)])

    return time_delay_embedded
