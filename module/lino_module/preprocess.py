import os
import time
import pickle
import pathlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import TensorDataset

from module.lino_module.model import WithAuxiliary

from typing import Tuple, Dict, Optional, Union, Callable
from pandas import DataFrame, Series
from numpy import ndarray
from torch import Tensor
from torch.utils.data import DataLoader
from colorama import Fore, Style


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


# ########################### 多変量拡張データセット ###############################
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
    """多次元データセットのメイン関数"""
    df = data.copy()
    if scaler is not None:
        data_index = data.index
        values = scaler().fit_transform(data.values.reshape(-1, 1))
        df[data_index] = values.reshape(-1)
    tded, label = delay_embeddings(df,
                                   seq,
                                   d_model,
                                   dilation,
                                   src_tgt_seq,
                                   step_num,
                                   daily, weekday, weekly, monthly)
    src, tgt = src_tgt_split(tded, *src_tgt_seq)
    train, test = to_torch_dataset(src, tgt, label, batch_size, train_rate)
    return train, test


def delay_embeddings(data: Series,
                     seq: int,
                     d_model: int,
                     dilation: int,
                     src_tgt_seq: Tuple[int],
                     step_num: int,
                     daily: bool, weekday: bool, weekly: bool, monthly: bool
                     ) -> Tuple[ndarray]:
    """TDEに対応した曜日、月時ラベルをconcatする"""
    # Time Delay Embedding
    index = data.index
    x, y = expand_and_split(data, seq, src_tgt_seq[1], step_num)
    tded, label = time_delay_embedding(x, y, d_model, dilation)

    # デイリーラベル
    if daily:
        scaled_day = (index.day - 1) / 31  # 0-1正規化
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


# ############################ 単変量データセット #################################
def univariate_dataset(data: Series,
                       seq: int,
                       dilation: int,
                       src_tgt_seq: Tuple[int],
                       step_num: int,
                       batch_size: int,
                       scaler: Optional[Union[StandardScaler, MinMaxScaler]],
                       daily: bool, weekday: bool, weekly:  bool, monthly: bool,
                       train_rate: float,
                       ) -> Tuple[DataLoader]:
    """単変量データセットのメイン関数"""
    df = data.copy()
    if scaler is not None:
        data_index = data.index
        values = scaler().fit_transform(data.values.reshape(-1, 1))
        df[data_index] = values.reshape(-1)

    x, y = concat_category(df,
                           seq,
                           dilation,
                           src_tgt_seq,
                           step_num,
                           daily, weekday, weekly, monthly)
    src, tgt = src_tgt_split(x, *src_tgt_seq)
    train, test = to_torch_dataset(src, tgt, y, batch_size, train_rate)
    return train, test


def concat_category(df: Series,
                    seq: int,
                    dilation: int,
                    src_tgt_seq: Tuple[int],
                    step_num: int,
                    daily: bool, weekday: bool, weekly: bool, monthly: bool
                    ) -> Tuple[ndarray]:
    tgt_seq = src_tgt_seq[1]
    x, y = uni_split(df, seq, dilation, tgt_seq, step_num)
    x = np.expand_dims(x, 1)

    index = df.index
    if daily:
        scaled_day = (index.day - 1) / 31  # 0-1正規化
        inf_day, _ = uni_split(scaled_day, seq, dilation, tgt_seq, step_num)
        inf_day = np.expand_dims(inf_day, 1)
        x = np.concatenate((x, inf_day), axis=1)

    if weekday:
        scaled_weekday = index.weekday / 6  # 0-1正規化
        inf_weekday, _ = uni_split(scaled_weekday, seq, dilation, tgt_seq, step_num)
        inf_weekday = np.expand_dims(inf_weekday, 1)
        x = np.concatenate((x, inf_weekday), axis=1)

    if weekly:
        scaled_week_num = (index.isocalendar().week - 1) / 44
        inf_weekly, _ = uni_split(scaled_week_num, seq, dilation, tgt_seq, step_num)
        inf_weekly = np.expand_dims(inf_weekly, 1)
        x = np.concatenate((x, inf_weekly), axis=1)

    if monthly:
        scaled_month = (index.month - 1) / 11
        inf_monthly, _ = uni_split(scaled_month, seq, dilation, tgt_seq, step_num)
        inf_monthly = np.expand_dims(inf_monthly, 1)
        x = np.concatenate((x, inf_monthly), axis=1)

    return x.transpose(0, 2, 1), y


def uni_split(ds: Series, seq: int, dilation: int, tgt_seq: Tuple[int],
              step_num: int) -> Tuple[ndarray]:
    data = ds.copy().values
    x_range = seq * (dilation + 1)
    num = len(data) - x_range - step_num + 1
    x = np.array([data[i: i + x_range: dilation + 1] for i in range(num)])

    y_start = x_range - dilation - tgt_seq + step_num
    y_end = x_range - dilation + step_num
    y = np.array([data[i + y_start: i + y_end] for i in range(num)])
    return x, y


# ######################### for inference ##########################
def model_loader(dir_path: str, model: Callable[[Tensor], Tensor]
                 ) -> Tuple[Dict[str, str], Callable[[Tensor], Tensor], str]:
    """再帰的推論関数用の呼び出し関数"""
    # ファイル情報の取得
    files = list(pathlib.Path(dir_path).glob('*'))
    for idx, i in enumerate(files):
        print(f'Index: {idx}')
        print(str(i.name))
        print()

    # インデックスの選択
    time.sleep(0.5)
    index = int(input('select model index'))

    file_name = os.path.splitext(str(files[index].name))[0]
    kw_path = dir_path + 'kw_inf/' + file_name + '.pkl'
    print(Fore.YELLOW + f'selected model :\n{file_name}' + Style.RESET_ALL)

    # 訓練時引数の読み込み
    with open(kw_path, 'rb') as f:
        kwrgs = pickle.load(f)

    # モデルパラメータの上書き
    file_path = str(files[index])
    model_kw = kwrgs['model']
    ride_model = model(**model_kw)
    ride_model.load_state_dict(torch.load(file_path))

    return kwrgs['dataset'], ride_model, file_name


def auxiliary_model_loader(dir_path, base_model, auxiliary_model):
    """再帰的推論関数用の呼び出し関数"""
    # ファイル情報の取得
    files = list(pathlib.Path(dir_path).glob('*'))
    for idx, i in enumerate(files):
        print(f'Index: {idx}')
        print(str(i.name))
        print()

    # インデックスの選択
    time.sleep(0.5)
    index = int(input('select model index'))
    file_name = os.path.splitext(str(files[index].name))[0]
    kw_path = dir_path + 'kw_inf/' + file_name + '.pkl'
    print(Fore.YELLOW + f'selected model :\n{file_name}' + Style.RESET_ALL)

    # 訓練時引数の読み込み
    with open(kw_path, 'rb') as f:
        kwrgs = pickle.load(f)

    # モデルパラメータの上書き
    file_path = str(files[index])

    base_model = base_model(**kwrgs['base_model'])
    auxiliary_model = auxiliary_model(**kwrgs['auxiliary_model'])
    ride_model = WithAuxiliary(base_model, auxiliary_model)
    ride_model.load_state_dict(torch.load(file_path))

    return kwrgs['dataset'], ride_model, file_name

