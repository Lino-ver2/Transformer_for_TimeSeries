# トレンド成分を削除する --> 定常データにしていく
from pandas import Series as Series

# 階差データを作る関数
def difference(dataset, interval = 1):

    # 階差データを格納するためのリストを用意
    diff = list()
    
    # 指定した階差の数からデータセットのサイズまでのループ
    for i in range(interval, len(dataset)):
        # 階差データの作成
        value = dataset[i] - dataset[i - interval]
        # 作成した階差データを準備しておいたリストに格納
        diff.append(value)

    return Series(diff)

# invert difference forecast --> 差分予測の反転？？？ --> 使途不明関数
def inverse_difference(last_ob, value):
    return value + last_ob