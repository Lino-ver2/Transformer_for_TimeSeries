import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import TensorDataset

from typing import Tuple, Optional, Union
from pandas import DataFrame, Series
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


def mode_of_freq(data: DataFrame,
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


def mode_of_freq(data: DataFrame,
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


def tde_dataset_wm(data: Series,
                   seq: int,
                   d_model: int,
                   dilation: int,
                   src_tgt_seq: Tuple[int],
                   step_num: int,
                   batch_size: int,
                   scaler: Optional[Union[StandardScaler, MinMaxScaler]],
                   daily: bool,
                   weekday: bool,
                   weekly:  bool,
                   monthly: bool,
                   train_rate: float,
                   ) -> Tuple[DataLoader]:
    """TDEに対応した曜日ラベルと月ラベル付与したデータセットのメイン関数"""
    df = data.copy()
    if scaler is not None:
        data_index = data.index
        values = scaler().fit_transform(data.values.reshape(-1, 1))
        df[data_index] = values.reshape(-1)
    tded, label = delay_embeddings(df,
                                   d_model,
                                   dilation,
                                   seq,
                                   src_tgt_seq,
                                   step_num,
                                   daily, weekday, weekly, monthly)
    src, tgt = src_tgt_split(tded, *src_tgt_seq)
    train, test = to_torch_dataset(src, tgt, label, batch_size, train_rate)
    return train, test


def delay_embeddings(data: Series, d_model: int, dilation: int, seq: int,
                     src_tgt_seq: Tuple[int], step_num: int,
                     daily: bool, weekday: bool, weekly: bool, monthly: bool
                     ) -> Tuple[ndarray]:
    """TDEに対応した曜日、月時ラベルをconcatする"""
    # Time Delay Embedding
    index = data.index
    x, y = expand_and_split(data, seq, src_tgt_seq[1], step_num)
    tded, label = time_delay_embedding(x, y, d_model, dilation)

    # デイリーラベル
    if daily:
        scaled_day = index.day / 31  # 0-1正規化
        day, _ = expand_and_split(scaled_day, seq, src_tgt_seq[1], step_num)
        tded_day = time_delay_embedding(day, None, d_model, dilation)
        tded = np.concatenate((tded, tded_day), axis=2)

    # 曜日ラベル
    if weekday:
        scaled_weekday = index.weekday / 6  # 0-1正規化
        week, _ = expand_and_split(scaled_weekday, seq, src_tgt_seq[1], step_num)
        tded_week = time_delay_embedding(week, None, d_model, dilation)
        tded = np.concatenate((tded, tded_week), axis=2)

    # 週次ラベル
    if weekly:
        scaled_week_num = (index.isocalendar().week - 1) / 44  # 0-1正規化
        week_num, _ = expand_and_split(scaled_week_num, seq, src_tgt_seq[1], step_num)
        tded_week_num = time_delay_embedding(week_num, None, d_model, dilation)
        tded = np.concatenate((tded, tded_week_num), axis=2)

    # 月ラベル
    if monthly:
        scaled_month = (index.month - 1) / 11  # 0-1正規化
        month, _ = expand_and_split(scaled_month, seq, src_tgt_seq[1], step_num)
        tded_month = time_delay_embedding(month, None, d_model, dilation)
        tded = np.concatenate((tded, tded_month), axis=2)
    return tded, label


def expand_and_split(ds: Series,
                     seq: int,
                     tgt_seq: int,
                     step_num: int
                     ) -> Tuple[ndarray]:
    """2次元にd_modelずらしたデータと正解データを作成する
    引数:
        ds: 単変量時系列データ
        seq: transformerのシーケンス
    """
    endpoint = len(ds) - (seq + tgt_seq + 1)
    expanded = np.stack([ds[i: i + seq + step_num] for i in range(0, endpoint)])
    x = expanded[:, :-step_num]
    y = expanded[:, -tgt_seq:]
    return x, y


def time_delay_embedding(x: ndarray, y: Optional[ndarray],
                         d_model: int, dilation: int) -> Tuple[ndarray]:
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


def src_tgt_split(tded: ndarray, src_seq: int, tgt_seq: int) -> Tuple[ndarray]:
    """エンコーダ入力とデコーダ入力への分割"""
    # 推論時
    if tded.ndim == 2:
        src = tded[:, :src_seq]
        tgt = tded[:, -tgt_seq:]
        return src.T, tgt.T
    # 訓練時（バッチ対応）
    if tded.ndim == 3:
        src = tded[:, :src_seq]
        tgt = tded[:, -tgt_seq:]
        return src, tgt


def to_torch_dataset(src: ndarray, tgt: ndarray, label: ndarray,
                     batch_size: int, train_rate: float) -> DataLoader:
    """Pytorch用のデータセットへの変換
    引数:
        src: エンコーダ入力データ
        tgt: デコーダ入力データ
        label: 正解データ
        batch_size: ミニバッチのバッチサイズ
    """
    if label.ndim == 1:
        label = label.reshape(-1, 1)[:len(src)]
    if label.ndim == 2:
        label = label[:len(src)]
    pack = (src, tgt, label)
    train_pack = [
        torch.from_numpy(i.astype(np.float32))[: int(len(src) * train_rate)]
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