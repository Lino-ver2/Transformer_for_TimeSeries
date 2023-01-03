import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset

from typing import Tuple, Optional
from pandas import DataFrame, Series, DatetimeIndex
from numpy import ndarray
from torch.utils.data import DataLoader


# 「mps」ではTransfomerのattnでエラーが出る
def select_device():
    """GPU もしくは CPU の選択"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('cuda is selected as device!')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('mps is selected as device!')
    else:
        device = torch.device('cpu')
        print('cpu....f')
    return device


def time_series_dataset(data: DataFrame,
                        seq: int,
                        d_model: int,
                        dilation: int,
                        src_tgt_seq: Tuple[int],
                        batch_size: int,
                        trg_column='item_cnt_day') -> Tuple[DataLoader]:
    """TDEデータセットのメイン関数"""
    data = getattr(_mode_of_freq(data), trg_column)
    data = StandardScaler().fit_transform(data.values.reshape(-1, 1))
    data = data.reshape(-1)
    x, y = _expand_and_split(data, seq)
    tded, label = _time_delay_embedding(x, y, d_model, dilation)
    src, tgt = _src_tgt_split(tded, *src_tgt_seq)
    train, test = _to_torch_dataset(src, tgt, label, batch_size)
    return train, test


def weekly_monthly_tde_dataset(data: DataFrame,
                               seq: int,
                               d_model: int,
                               dilation: int,
                               src_tgt_seq: Tuple[int],
                               batch_size: int,
                               trg_column='item_cnt_day') -> Tuple[DataLoader]:
    """TDEに対応した曜日ラベルと月ラベル付与したデータセットのメイン関数"""
    data = getattr(_mode_of_freq(data), trg_column)
    index = data.index
    data = StandardScaler().fit_transform(data.values.reshape(-1, 1))
    data = data.reshape(-1)
    x, y = _expand_and_split(data, seq)
    tded, label = _delay_embeddings(
                                x, y,
                                index,
                                d_model,
                                dilation,
                                seq,
                                weekly=True, monthly=True)
    src, tgt = _src_tgt_split(tded, *src_tgt_seq)
    train, test = _to_torch_dataset(src, tgt, label, batch_size)
    return train, test


def _mode_of_freq(data: DataFrame,
                  key='date',
                  freq='D',
                  mode='sum'
                  ) -> DataFrame:
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


def _expand_and_split(ds: Series, seq: int) -> Tuple[ndarray]:
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


def _time_delay_embedding(x: ndarray,
                          y: Optional[ndarray],
                          d_model: int,
                          dilation: int) -> Tuple[ndarray]:
    """Time Delay Embedding
    引数:
        x: 訓練データ
        y: 正解データ
        d_model: エンべディング次元数
        dilation: エンべディングの間隔
    """
    endpoint = x.shape[0] - d_model * (dilation + 1)
    span = d_model * (dilation + 1)

    tded = [x[i: i + span: (dilation + 1), :].T for i in range(endpoint)]
    if y is not None:
        y = y[span - (dilation + 1):]
        return np.array(tded), np.array(y)
    return np.array(tded)


def _src_tgt_split(tded: ndarray,
                   src_seq: int,
                   tgt_seq: int) -> Tuple[ndarray]:
    """エンコーダ入力とデコーダ入力への分割"""
    src = tded[:, :src_seq]
    tgt = tded[:, -tgt_seq:]
    return src, tgt


def _delay_embeddings(x: ndarray,
                      y: ndarray,
                      index: DatetimeIndex,
                      d_model: int,
                      dilation: int,
                      seq: int,
                      weekly=True,
                      monthly=True):
    """TDEに対応した曜日、月時ラベルをconcatする"""
    # Time Delay Embedding
    tded, label = _time_delay_embedding(x, y, d_model, dilation)

    # 曜日ラベル
    if weekly:
        # positional encodingのために0-1でスケーリング
        weekly_num = list(np.linspace(0, 1, 7))
        # 曜日ラベルをデータ数分ループさせたシーケンス
        weekly_label = weekly_num * (len(index) // 7) + weekly_num[:len(index) % 7]
        week, _ = _expand_and_split(weekly_label, seq)
        tded_week = _time_delay_embedding(week, None, d_model, dilation)
        tded = np.concatenate((tded, tded_week), axis=2)

    # 月ラベル
    if monthly:
        # positional encodingのために0-1でスケーリング
        scaled_index = (index.month - 1) / 11
        month, _ = _expand_and_split(scaled_index, seq)
        tded_month = _time_delay_embedding(month, None, d_model, dilation)
        tded = np.concatenate((tded, tded_month), axis=2)
    return tded, label


def _to_torch_dataset(src: ndarray,
                      tgt: ndarray,
                      label: ndarray,
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
        torch.from_numpy(i.astype(np.float32))[:int(len(src) * train_rate)]
        for i in pack
        ]
    test_pack = [
        torch.from_numpy(i.astype(np.float32))[int(len(src) * train_rate):]
        for i in pack
        ]
    train = TensorDataset(*train_pack)
    train = DataLoader(train, batch_size, shuffle=False)
    test = TensorDataset(*test_pack)
    test = DataLoader(test, batch_size=1, shuffle=False)
    return train, test
