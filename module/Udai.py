import pandas as pd
from typing import List, Tuple

# Visualize
import matplotlib.pyplot as plt

# TIME SERIES
from pandas.plotting import autocorrelation_plot
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs


# 階差データを作る関数
def difference(dataset: pd.Series, 
               interval = 1) -> pd.Series:

    # 階差データを格納するためのリストを用意
    diff = list()
    
    # 指定した階差の数からデータセットのサイズまでのループ
    for i in range(interval, len(dataset)):
        # 階差データの作成
        value = dataset[i] - dataset[i - interval]
        # 作成した階差データを準備しておいたリストに格納
        diff.append(value)

    return pd.Series(diff)

# invert difference forecast --> 差分予測の反転？？？ --> 使途不明関数
def inverse_difference(last_ob, value):
    return value + last_ob

# ADF検定
def test_stationarity(timeseries: pd.Series):

    # ADF検定の実行
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag = 'AIC')

    # 結果の格納
    dfoutput = pd.Series(dftest[0:4], 
                         index = ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    
    # ADF検定量における、各棄却域の表示
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

# グラフ可視化の関数を作成する
def tsplot(y: pd.Series, 
           lags = None, 
           figsize = (10, 8), 
           style = 'bmh', 
           title = '') -> plt.show(3,2):

    # yがpandasのSeries型でなければ、起動する。
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # 描画する際の設定を呼び出す。with文を使用するので、この関数の使用時のみ、呼び出すことになる。
    with plt.style.context(style):
        # サイズ
        fig = plt.figure(figsize = figsize)
        # 計５種類のグラフを設定
        layout = (3,2)
        ts_ax = plt.subplot2grid(layout, (0,0), colspan = 2) # 原系列データ
        acf_ax = plt.subplot2grid(layout, (1,0)) # acfのコレログラム
        pacf_ax = plt.subplot2grid(layout, (1,1)) # pacfのコレログラム
        qq_ax = plt.subplot2grid(layout, (2,0)) # qqプロット。quantile-quantile。二つの確率分布を比較。(正規分布と〇〇) https://mathwords.net/qqplot 
        pp_ax = plt.subplot2grid(layout, (2,1)) # ppプロット。累積確率値をプロットし、2つの分布の類似度を比較。

        y.plot(ax = ts_ax)
        ts_ax.set_title(title)

        smt.graphics.plot_acf(y, lags = lags, ax = acf_ax, alpha = 0.5)

        smt.graphics.plot_pacf(y, lags = lags, ax = pacf_ax, alpha = 0.5)

        sm.qqplot(y, line = "s", ax = qq_ax)
        qq_ax.set_title("QQ Plot")
        
        scs.probplot(y, sparams = (y.mean(), y.std()), plot = pp_ax)

        plt.tight_layout()

    return