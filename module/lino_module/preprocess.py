from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader


# GPU もしくは CPU の選択
def select_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('cuda is selected as device!')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('mps is selected as device!')
    else:
        device = torch.device('cpu')
    return device


# データセットの main関数
def time_series_dataset(data: pd.DataFrame,
                        trg_column='item_cnt_day',
                        seq=7,
                        d_model=32,
                        dilation=1,
                        src_tgt_seq=(6, 2),
                        batch_size=64) -> Tuple[DataLoader]:
    data = getattr(_mode_of_freq(data), trg_column)
    data = StandardScaler().fit_transform(data.values.reshape(-1, 1))
    data = data.reshape(-1)
    x, y = _expand_and_split(data, seq)
    tded, label = _time_delay_embedding(x, y, d_model, dilation)
    src, tgt = _src_tgt_split(tded, *src_tgt_seq)
    train, test = _to_torch_dataset(src, tgt, label, batch_size)
    return train, test


def _mode_of_freq(data: pd.DataFrame,
                  key='date',
                  freq='D',
                  mode='sum'
                  ) -> pd.DataFrame:
    """時系列データを基本統計量で統合する
    引数:
        data: 対象を含むオリジナルデータ
        key: 時間軸のカラム名
        freq: グループ単位（D: 日ごと, M: 月ごと, Y: 年ごと）
        mode: 統計量（sum, mean, etc）
    """
    # 日付をobjectからdate_time型に変更
    data[key] = pd.to_datetime(data[key], format=('%d.%m.%Y'))
    # 時系列(key)についてグループ単位(freq)の売上数の基本統計量(mode)で出力
    mode_of_key = getattr(data.groupby(pd.Grouper(key=key, freq=freq)), mode)
    return mode_of_key()


def _expand_and_split(ds: pd.Series, seq: int) -> Tuple[np.ndarray]:
    """2次元にd_modelずらしたデータと正解データを作成する
    引数:
        ds: 単変量時系列データ
        seq: transformerのシーケンス
    """
    endpoint = len(ds) - (seq + 1)
    expanded = np.stack([ds[i: i + seq + 1] for i in range(0, endpoint)])
    x = expanded[:, :-1]
    y = expanded[:, -1]
    return x, y  # ,expanded  # 挙動の確認用


def _time_delay_embedding(x: np.ndarray,
                          y: np.ndarray,
                          d_model=32,
                          dilation=1) -> Tuple[np.ndarray]:
    """Time Delay Embedding
    引数:
        x: 訓練データ
        y: 正解データ
        d_model: エンべディング次元数
        dilation: エンべディングの膨張率 
    """
    endpoint = x.shape[0] - d_model * dilation
    span = d_model * dilation

    tded = [x[i: i + span: dilation, :].T for i in range(endpoint)] 
    y = y[span - dilation:]
    return np.array(tded), np.array(y)

## time_delay_embeddingの挙動確認用（メモ）
# i = 0
# print(expanded[i: i + span: dilation, :][-1,   -2:])
# print(tded[i][-1, -1], y_[i])


def _src_tgt_split(tded: np.ndarray,
                   src_seq: int,
                   tgt_seq: int) -> Tuple[np.ndarray]:
    """エンコーダ入力とデコーダ入力への分割"""
    src = tded[:, :src_seq]
    tgt = tded[:, -tgt_seq:]
    return src, tgt


def _to_torch_dataset(src: np.ndarray,
                      tgt: np.ndarray,
                      label: np.ndarray,
                      batch_size: int,
                      train_rate=0.9) -> DataLoader:
    """Pytorch用のデータセットへの変換
    引数:
        src: エンコーダ入力データ
        tgt: デコーダ入力データ
        label: 正解データ
        batch_size: ミニバッチのバッチサイズ
    """
    label = label.reshape(-1, 1)[:len(src)]
    pack = (src, tgt, label)
    train_pack = [
        torch.from_numpy(i.astype(np.float32)).clone()[:int(len(src) * train_rate)]
        for i in pack
        ]
    test_pack = [
        torch.from_numpy(i.astype(np.float32)).clone()[int(len(src) * train_rate):]
        for i in pack
        ]
    train = TensorDataset(*train_pack)
    train = DataLoader(train, batch_size, shuffle=False)
    test = TensorDataset(*test_pack)
    test = DataLoader(test, batch_size=1, shuffle=False)
    return train, test
